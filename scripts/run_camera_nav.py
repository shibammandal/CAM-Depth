"""
Run camera-based navigation demo.
Demonstrates navigation using stereo camera depth estimation.
Can optionally use a trained RL model to optimize depth parameters.
"""

import argparse
import os
import sys
import time
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from simulation.world import SimulationWorld
from simulation.robot import DifferentialDriveRobot
from simulation.sensors import StereoCamera
from depth.stereo_matching import StereoDepthEstimator
from navigation.camera_nav import CameraNavigator
from visualization.live_display import LiveDisplay

try:
    from stable_baselines3 import PPO, SAC
except ImportError:
    print("Stable Baselines3 not found. RL support disabled.")


def parse_args():
    parser = argparse.ArgumentParser(description='Run camera navigation demo')
    parser.add_argument('--episodes', type=int, default=3,
                       help='Number of episodes (default: 3)')
    parser.add_argument('--max-steps', type=int, default=500,
                       help='Max steps per episode (default: 500)')
    parser.add_argument('--visualize', action='store_true',
                       help='Show matplotlib visualization')
    parser.add_argument('--headless', action='store_true',
                       help='Run without PyBullet GUI')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained RL model (zip file)')
    return parser.parse_args()


def safe_metric(val, default=10.0):
    """Clip inf values to prevent NaN in observation."""
    if val is None or not np.isfinite(val):
        return default
    return float(val)


def get_observation(depth_estimator, metrics, robot_pos, scene_idx=0):
    """Construct observation vector for RL agent."""
    # Current parameters (normalized)
    params = depth_estimator.get_param_vector()
    
    # Depth quality metrics
    metrics_vec = np.array([
        safe_metric(metrics.get('mae', 10.0), 10.0),
        safe_metric(metrics.get('valid_ratio', 0.0), 0.0),
        safe_metric(metrics.get('accuracy_10pct', 0.0), 0.0),
        safe_metric(metrics.get('rmse', 10.0), 10.0),
        safe_metric(metrics.get('smoothness', 0.0), 0.0),
    ], dtype=np.float32)
    
    # Scene info
    scene_info = np.array([
        robot_pos[0] / 10.0,
        robot_pos[1] / 10.0,
        scene_idx,
    ], dtype=np.float32)
    
    return np.concatenate([params, metrics_vec, scene_info])


def main():
    args = parse_args()
    
    print("=" * 60)
    print("CAM-Depth: Stereo Camera Navigation Demo")
    print("=" * 60)
    
    # Load model if provided
    model = None
    if args.model:
        print(f"Loading model from: {args.model}")
        try:
            # Try loading as PPO first, then SAC
            try:
                model = PPO.load(args.model)
                print("Loaded PPO model")
            except:
                model = SAC.load(args.model)
                print("Loaded SAC model")
        except Exception as e:
            print(f"Failed to load model: {e}")
            return
    
    # Initialize simulation
    print("\nInitializing simulation...")
    world = SimulationWorld(args.config, gui=not args.headless)
    world.initialize()
    world.create_walls()
    
    robot = DifferentialDriveRobot(args.config)
    stereo_camera = StereoCamera(args.config)
    
    # Initialize depth estimator
    depth_estimator = StereoDepthEstimator(args.config)
    intrinsics = stereo_camera.get_intrinsics()
    depth_estimator.set_camera_params(intrinsics['fx'], intrinsics['baseline'])
    
    navigator = CameraNavigator()
    
    # Setup visualization
    display = None
    if args.visualize:
        display = LiveDisplay()
        display.setup()
    
    # Run episodes
    for ep in range(args.episodes):
        print(f"\n--- Episode {ep + 1}/{args.episodes} ---")
        
        # Random start and goal
        start_pos = np.array([np.random.uniform(-7, -3), np.random.uniform(-7, -3)])
        goal_pos = np.array([np.random.uniform(3, 7), np.random.uniform(3, 7)])
        
        print(f"Start: ({start_pos[0]:.1f}, {start_pos[1]:.1f})")
        print(f"Goal:  ({goal_pos[0]:.1f}, {goal_pos[1]:.1f})")
        
        # Setup scene
        world.reset()
        world.create_random_obstacles(10, 2.0, tuple(start_pos), tuple(goal_pos))
        world.set_markers(tuple(start_pos), tuple(goal_pos))
        
        robot.spawn(tuple(start_pos), 0)
        
        depth_errors = []
        last_metrics = {}  # For RL observation
        
        # Navigation loop
        for step in range(args.max_steps):
            # Get robot state
            pos = robot.get_position()
            heading = robot.get_orientation()
            
            # --- RL Inference Step ---
            if model:
                # Construct observation
                obs = get_observation(depth_estimator, last_metrics, pos)
                # Predict action
                action, _ = model.predict(obs, deterministic=True)
                # Apply action (update parameters)
                depth_estimator.set_param_vector(action)
            
            # Get stereo images
            left_pos, right_pos, orientation = robot.get_camera_positions()
            left_rgb, right_rgb, left_gt, _ = stereo_camera.capture_stereo(
                left_pos, right_pos, orientation,
                use_opengl=not args.headless
            )
            
            # Estimate depth
            estimated_depth, metrics = depth_estimator.compute_depth_with_quality(
                left_rgb, right_rgb, left_gt
            )
            last_metrics = metrics  # Update for next step
            
            if 'mae' in metrics and np.isfinite(metrics['mae']):
                depth_errors.append(metrics['mae'])
            
            # Check goal
            dist_to_goal = np.linalg.norm(pos[:2] - goal_pos)
            if dist_to_goal < 0.5:
                print(f"✓ Goal reached in {step} steps!")
                if depth_errors:
                    print(f"  Average depth error: {np.mean(depth_errors):.3f}m")
                break
            
            # Compute velocity using depth
            linear, angular = navigator.compute_velocity(
                pos, heading, goal_pos, estimated_depth, stereo_camera.fov
            )
            
            # Apply velocity
            robot.set_velocity(linear, angular)
            
            # Step simulation
            for _ in range(10):
                world.step()
            
            # Update display
            if display and step % 5 == 0:
                display.update(
                    left_image=left_rgb,
                    right_image=right_rgb,
                    depth_map=estimated_depth,
                    gt_depth=left_gt,
                    robot_pos=pos,
                    goal_pos=goal_pos,
                    depth_error=metrics.get('mae', 0),
                    params=depth_estimator.get_params()
                )
            
            # Small delay for visualization
            if not args.headless:
                time.sleep(0.02)
        
        else:
            print(f"✗ Did not reach goal within {args.max_steps} steps")
            if depth_errors:
                print(f"  Average depth error: {np.mean(depth_errors):.3f}m")
        
        robot.remove()
        if display:
            display.reset()
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
    
    # Cleanup
    if display:
        display.close()
    world.close()


if __name__ == '__main__':
    main()
