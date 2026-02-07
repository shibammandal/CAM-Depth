"""
Gymnasium environment for depth parameter optimization using RL.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Optional, Any
import yaml

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.world import SimulationWorld
from simulation.robot import DifferentialDriveRobot
from simulation.sensors import LidarSensor, StereoCamera
from depth.stereo_matching import StereoDepthEstimator
from rl.rewards import depth_error_reward, smoothness_reward


class DepthOptimizationEnv(gym.Env):
    """
    Gymnasium environment for RL-based stereo depth parameter optimization.
    
    The agent learns to adjust stereo matching parameters to minimize
    depth estimation error compared to ground truth.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    def __init__(self, 
                 config_path: str = "configs/config.yaml",
                 render_mode: Optional[str] = None,
                 headless: bool = False):
        """
        Initialize the environment.
        
        Args:
            config_path: Path to configuration file
            render_mode: Rendering mode ('human' or 'rgb_array')
            headless: Run simulation without GUI
        """
        super().__init__()
        
        self.config_path = config_path
        self.render_mode = render_mode
        self.headless = headless
        
        # Load config
        self.config = self._load_config(config_path)
        
        # Initialize components (lazy loading)
        self.world = None
        self.robot = None
        self.stereo_camera = None
        self.depth_estimator = None
        self._initialized = False
        
        # Episode state
        self.step_count = 0
        self.max_steps = self.config.get('navigation', {}).get('max_episode_steps', 100)
        self.current_scene_idx = 0
        self.num_scenes_per_episode = 10  # Number of different viewpoints per episode
        
        # Goal position (randomized each reset)
        self.goal_pos = np.array([5.0, 5.0])
        self.start_pos = np.array([-5.0, -5.0])
        
        # Define action space: normalized stereo matching parameters
        # Each parameter in [0, 1] range, will be denormalized when applied
        param_dim = StereoDepthEstimator.param_dim()
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(param_dim,), dtype=np.float32
        )
        
        # Define observation space
        # Option 1: Just current parameters + depth quality metrics
        # Option 2: Include stereo images (more complex)
        # We'll use Option 1 for efficiency
        
        # Observation: [current_params (9) + depth_metrics (5) + scene_info (3)]
        obs_dim = param_dim + 5 + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Tracking
        self.last_depth_metrics = {}
        self.episode_rewards = []
        self.best_params = None
        self.best_reward = -float('inf')
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {}
    
    def _initialize_simulation(self):
        """Lazy initialization of simulation components."""
        if self._initialized:
            return
        
        # Create world
        self.world = SimulationWorld(self.config_path, gui=not self.headless)
        self.world.initialize()
        self.world.create_walls()
        
        # Create robot
        self.robot = DifferentialDriveRobot(self.config_path)
        self.robot.spawn(position=tuple(self.start_pos), orientation=0)
        
        # Create sensors
        self.stereo_camera = StereoCamera(self.config_path)
        
        # Create depth estimator
        self.depth_estimator = StereoDepthEstimator(self.config_path)
        intrinsics = self.stereo_camera.get_intrinsics()
        self.depth_estimator.set_camera_params(
            focal_length=intrinsics['fx'],
            baseline=intrinsics['baseline']
        )
        
        self._initialized = True
    
    def _create_random_scene(self):
        """Create a random obstacle scene."""
        self.world.reset()
        
        # Random start and goal positions
        half_size = self.world.world_size[0] / 2 - 2
        self.start_pos = np.array([
            np.random.uniform(-half_size, -half_size/2),
            np.random.uniform(-half_size, -half_size/2)
        ])
        self.goal_pos = np.array([
            np.random.uniform(half_size/2, half_size),
            np.random.uniform(half_size/2, half_size)
        ])
        
        # Create obstacles
        self.world.create_random_obstacles(
            num_obstacles=np.random.randint(5, 15),
            exclude_radius=2.0,
            start_pos=tuple(self.start_pos),
            goal_pos=tuple(self.goal_pos)
        )
        
        # Set markers
        self.world.set_markers(tuple(self.start_pos), tuple(self.goal_pos))
        
        # Reset robot position
        self.robot.reset(position=tuple(self.start_pos), orientation=0)
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation.
        
        Returns:
            Observation vector
        """
        # Current parameters (normalized)
        params = self.depth_estimator.get_param_vector()
        
        # Depth quality metrics (from last estimation)
        # Clip inf values to large but finite numbers to prevent NaN
        def safe_metric(val, default=10.0):
            if val is None or not np.isfinite(val):
                return default
            return float(val)
        
        metrics = np.array([
            safe_metric(self.last_depth_metrics.get('mae', 10.0), 10.0),
            safe_metric(self.last_depth_metrics.get('valid_ratio', 0.0), 0.0),
            safe_metric(self.last_depth_metrics.get('accuracy_10pct', 0.0), 0.0),
            safe_metric(self.last_depth_metrics.get('rmse', 10.0), 10.0),
            safe_metric(self.last_depth_metrics.get('smoothness', 0.0), 0.0),
        ], dtype=np.float32)
        
        # Scene info
        robot_pos = self.robot.get_position()
        scene_info = np.array([
            robot_pos[0] / 10.0,  # Normalized position
            robot_pos[1] / 10.0,
            self.current_scene_idx / self.num_scenes_per_episode,
        ], dtype=np.float32)
        
        return np.concatenate([params, metrics, scene_info])
    
    def _compute_depth_and_metrics(self) -> Tuple[np.ndarray, Dict]:
        """
        Capture stereo images, compute depth, and evaluate quality.
        
        Returns:
            (depth_map, metrics)
        """
        # Get camera positions
        left_pos, right_pos, orientation = self.robot.get_camera_positions()
        
        # Capture stereo images
        left_rgb, right_rgb, left_gt, right_gt = self.stereo_camera.capture_stereo(
            left_pos, right_pos, orientation,
            use_opengl=not self.headless
        )
        
        # Get ground truth depth (from left camera view)
        ground_truth = left_gt
        
        # Compute stereo depth with current parameters
        estimated_depth, quality_metrics = self.depth_estimator.compute_depth_with_quality(
            left_rgb, right_rgb, ground_truth
        )
        
        # Add smoothness metric
        quality_metrics['smoothness'] = smoothness_reward(estimated_depth)
        
        return estimated_depth, quality_metrics
    
    def reset(self, seed: Optional[int] = None, 
              options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment.
        
        Args:
            seed: Random seed
            options: Reset options
            
        Returns:
            (observation, info)
        """
        super().reset(seed=seed)
        
        # Initialize simulation if needed
        self._initialize_simulation()
        
        # Create new random scene
        self._create_random_scene()
        
        # Reset episode state
        self.step_count = 0
        self.current_scene_idx = 0
        self.episode_rewards = []
        
        # Reset depth estimator to random parameters
        random_params = np.random.uniform(0, 1, size=StereoDepthEstimator.param_dim())
        self.depth_estimator.set_param_vector(random_params)
        
        # Compute initial depth to get metrics
        _, self.last_depth_metrics = self._compute_depth_and_metrics()
        
        obs = self._get_observation()
        info = {'metrics': self.last_depth_metrics}
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Normalized parameter vector [0, 1]^n
            
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        self.step_count += 1
        
        # Apply new parameters
        action = np.clip(action, 0, 1)
        self.depth_estimator.set_param_vector(action)
        
        # Move robot to new viewpoint (random position along path to goal)
        t = (self.current_scene_idx + 1) / self.num_scenes_per_episode
        new_pos = (1 - t) * self.start_pos + t * self.goal_pos
        # Add some randomness
        new_pos += np.random.randn(2) * 0.5
        self.robot.reset(position=tuple(new_pos), 
                        orientation=np.arctan2(self.goal_pos[1] - new_pos[1],
                                              self.goal_pos[0] - new_pos[0]))
        
        # Step simulation
        for _ in range(10):
            self.world.step()
        
        # Compute depth and metrics
        depth_map, metrics = self._compute_depth_and_metrics()
        self.last_depth_metrics = metrics
        
        # Calculate reward
        # Get ground truth depth for reward calculation
        left_pos, right_pos, orientation = self.robot.get_camera_positions()
        gt_depth = self.stereo_camera.get_ground_truth_depth(
            left_pos, orientation, use_opengl=not self.headless
        )
        reward, reward_metrics = depth_error_reward(depth_map, gt_depth)
        
        # Add smoothness bonus
        reward += metrics.get('smoothness', 0) * 0.5
        
        self.episode_rewards.append(reward)
        
        # Track best parameters
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_params = self.depth_estimator.get_params()
        
        # Episode termination
        self.current_scene_idx += 1
        terminated = False
        truncated = (self.step_count >= self.max_steps or 
                    self.current_scene_idx >= self.num_scenes_per_episode)
        
        # Get observation
        obs = self._get_observation()
        
        info = {
            'metrics': metrics,
            'reward_components': reward_metrics,
            'params': self.depth_estimator.get_params(),
            'best_params': self.best_params,
            'episode_reward': sum(self.episode_rewards),
        }
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment."""
        if self.render_mode == 'rgb_array':
            # Capture current view
            left_pos, right_pos, orientation = self.robot.get_camera_positions()
            left_rgb, right_rgb, _, _ = self.stereo_camera.capture_stereo(
                left_pos, right_pos, orientation,
                use_opengl=not self.headless
            )
            return left_rgb
        
        # Human mode is handled by PyBullet GUI
        return None
    
    def close(self):
        """Clean up resources."""
        if self.world is not None:
            self.world.close()
            self._initialized = False
