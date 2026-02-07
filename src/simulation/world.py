"""
PyBullet simulation world setup.
Handles physics simulation, obstacle creation, and goal markers.
"""

import pybullet as p
import pybullet_data
import numpy as np
from typing import Tuple, List, Optional
import yaml


class SimulationWorld:
    """PyBullet simulation environment for robot navigation."""
    
    def __init__(self, config_path: str = "configs/config.yaml", gui: bool = True):
        """
        Initialize the simulation world.
        
        Args:
            config_path: Path to configuration YAML file
            gui: Whether to enable GUI visualization
        """
        self.config = self._load_config(config_path)
        self.gui = gui
        self.physics_client = None
        self.ground_id = None
        self.obstacles = []
        self.goal_marker = None
        self.start_marker = None
        
        # World bounds
        self.world_size = self.config['simulation']['world_size']
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            # Return default config if file not found
            return {
                'simulation': {
                    'gui': True,
                    'timestep': 0.01,
                    'gravity': -9.81,
                    'world_size': [20.0, 20.0]
                }
            }
    
    def initialize(self) -> int:
        """
        Initialize PyBullet physics engine and create ground plane.
        
        Returns:
            Physics client ID
        """
        if self.gui:
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        # Set up physics
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, self.config['simulation']['gravity'])
        p.setTimeStep(self.config['simulation']['timestep'])
        
        # Create ground plane
        self.ground_id = p.loadURDF("plane.urdf")
        
        # Set camera for better viewing
        if self.gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=15,
                cameraYaw=45,
                cameraPitch=-45,
                cameraTargetPosition=[0, 0, 0]
            )
        
        return self.physics_client
    
    def create_walls(self) -> List[int]:
        """
        Create boundary walls around the world.
        
        Returns:
            List of wall body IDs
        """
        wall_ids = []
        half_size = [self.world_size[0] / 2, self.world_size[1] / 2]
        wall_height = 1.0
        wall_thickness = 0.2
        
        # Wall positions: [x, y, yaw]
        walls = [
            [0, half_size[1], 0, half_size[0], wall_thickness],  # Top
            [0, -half_size[1], 0, half_size[0], wall_thickness],  # Bottom
            [half_size[0], 0, np.pi/2, half_size[1], wall_thickness],  # Right
            [-half_size[0], 0, np.pi/2, half_size[1], wall_thickness],  # Left
        ]
        
        for x, y, yaw, length, thickness in walls:
            wall_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[length, thickness, wall_height]
            )
            wall_visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[length, thickness, wall_height],
                rgbaColor=[0.5, 0.5, 0.5, 1]
            )
            wall_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=wall_shape,
                baseVisualShapeIndex=wall_visual,
                basePosition=[x, y, wall_height],
                baseOrientation=p.getQuaternionFromEuler([0, 0, yaw])
            )
            wall_ids.append(wall_id)
        
        return wall_ids
    
    def add_obstacle(self, position: Tuple[float, float], 
                     size: Tuple[float, float, float] = (0.5, 0.5, 1.0),
                     color: Tuple[float, float, float, float] = (0.8, 0.2, 0.2, 1)) -> int:
        """
        Add a box obstacle to the world.
        
        Args:
            position: (x, y) position of obstacle
            size: (width, depth, height) of obstacle
            color: RGBA color
            
        Returns:
            Body ID of the obstacle
        """
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[size[0]/2, size[1]/2, size[2]/2]
        )
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[size[0]/2, size[1]/2, size[2]/2],
            rgbaColor=color
        )
        obstacle_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[position[0], position[1], size[2]/2]
        )
        self.obstacles.append(obstacle_id)
        return obstacle_id
    
    def add_cylinder_obstacle(self, position: Tuple[float, float],
                              radius: float = 0.3, height: float = 1.0,
                              color: Tuple[float, float, float, float] = (0.2, 0.6, 0.2, 1)) -> int:
        """
        Add a cylindrical obstacle to the world.
        
        Args:
            position: (x, y) position
            radius: Cylinder radius
            height: Cylinder height
            color: RGBA color
            
        Returns:
            Body ID
        """
        collision_shape = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=radius,
            height=height
        )
        visual_shape = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=radius,
            length=height,
            rgbaColor=color
        )
        obstacle_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[position[0], position[1], height/2]
        )
        self.obstacles.append(obstacle_id)
        return obstacle_id
    
    def create_random_obstacles(self, num_obstacles: int = 10, 
                                 exclude_radius: float = 2.0,
                                 start_pos: Tuple[float, float] = (0, 0),
                                 goal_pos: Tuple[float, float] = (0, 0)) -> List[int]:
        """
        Create random obstacles avoiding start and goal positions.
        
        Args:
            num_obstacles: Number of obstacles to create
            exclude_radius: Radius around start/goal to keep clear
            start_pos: Start position to avoid
            goal_pos: Goal position to avoid
            
        Returns:
            List of obstacle IDs
        """
        obstacle_ids = []
        half_size = [self.world_size[0] / 2 - 1, self.world_size[1] / 2 - 1]
        
        for _ in range(num_obstacles):
            while True:
                x = np.random.uniform(-half_size[0], half_size[0])
                y = np.random.uniform(-half_size[1], half_size[1])
                
                # Check distance from start and goal
                dist_start = np.sqrt((x - start_pos[0])**2 + (y - start_pos[1])**2)
                dist_goal = np.sqrt((x - goal_pos[0])**2 + (y - goal_pos[1])**2)
                
                if dist_start > exclude_radius and dist_goal > exclude_radius:
                    break
            
            # Randomly choose box or cylinder
            if np.random.random() > 0.5:
                size = (
                    np.random.uniform(0.3, 1.0),
                    np.random.uniform(0.3, 1.0),
                    np.random.uniform(0.5, 1.5)
                )
                obstacle_id = self.add_obstacle((x, y), size)
            else:
                radius = np.random.uniform(0.2, 0.5)
                height = np.random.uniform(0.5, 1.5)
                obstacle_id = self.add_cylinder_obstacle((x, y), radius, height)
            
            obstacle_ids.append(obstacle_id)
        
        return obstacle_ids
    
    def set_markers(self, start_pos: Tuple[float, float], 
                    goal_pos: Tuple[float, float]) -> Tuple[int, int]:
        """
        Create visual markers for start and goal positions.
        
        Args:
            start_pos: (x, y) start position
            goal_pos: (x, y) goal position
            
        Returns:
            (start_marker_id, goal_marker_id)
        """
        # Start marker (green sphere)
        start_visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.3,
            rgbaColor=[0, 1, 0, 0.7]
        )
        self.start_marker = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=start_visual,
            basePosition=[start_pos[0], start_pos[1], 0.3]
        )
        
        # Goal marker (blue sphere)
        goal_visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.3,
            rgbaColor=[0, 0, 1, 0.7]
        )
        self.goal_marker = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=goal_visual,
            basePosition=[goal_pos[0], goal_pos[1], 0.3]
        )
        
        # Add text labels if GUI
        if self.gui:
            p.addUserDebugText("START", [start_pos[0], start_pos[1], 1], 
                              textColorRGB=[0, 1, 0], textSize=1.5)
            p.addUserDebugText("GOAL", [goal_pos[0], goal_pos[1], 1],
                              textColorRGB=[0, 0, 1], textSize=1.5)
        
        return self.start_marker, self.goal_marker
    
    def get_ground_truth_depth(self, camera_pos: np.ndarray, 
                               camera_orientation: np.ndarray,
                               width: int = 320, height: int = 240,
                               fov: float = 60, near: float = 0.1, 
                               far: float = 10.0) -> np.ndarray:
        """
        Get ground truth depth map using PyBullet's depth buffer.
        This is used to compare against stereo-estimated depth.
        
        Args:
            camera_pos: Camera position [x, y, z]
            camera_orientation: Camera orientation quaternion
            width: Image width
            height: Image height
            fov: Field of view in degrees
            near: Near clipping plane
            far: Far clipping plane
            
        Returns:
            Depth map as numpy array
        """
        # Calculate view matrix
        rotation_matrix = np.array(p.getMatrixFromQuaternion(camera_orientation)).reshape(3, 3)
        forward = rotation_matrix @ np.array([1, 0, 0])
        up = rotation_matrix @ np.array([0, 0, 1])
        target = camera_pos + forward
        
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_pos.tolist(),
            cameraTargetPosition=target.tolist(),
            cameraUpVector=up.tolist()
        )
        
        # Projection matrix
        aspect = width / height
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=fov,
            aspect=aspect,
            nearVal=near,
            farVal=far
        )
        
        # Render image
        _, _, _, depth_buffer, _ = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL if self.gui else p.ER_TINY_RENDERER
        )
        
        # Convert depth buffer to actual depth values
        depth = far * near / (far - (far - near) * depth_buffer)
        
        return np.array(depth)
    
    def step(self):
        """Advance physics simulation by one timestep."""
        p.stepSimulation()
    
    def reset(self):
        """Reset simulation by removing all obstacles."""
        for obs_id in self.obstacles:
            p.removeBody(obs_id)
        self.obstacles = []
        
        if self.start_marker is not None:
            p.removeBody(self.start_marker)
            self.start_marker = None
        if self.goal_marker is not None:
            p.removeBody(self.goal_marker)
            self.goal_marker = None
    
    def close(self):
        """Disconnect from physics server."""
        if self.physics_client is not None:
            p.disconnect()
            self.physics_client = None
