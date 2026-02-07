"""
Differential drive robot model for PyBullet simulation.
Includes sensor mounting points and motor control.
"""

import pybullet as p
import numpy as np
from typing import Tuple, Optional
import yaml


class DifferentialDriveRobot:
    """A simple differential drive robot with sensor mounts."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize robot parameters from config.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        
        # Robot parameters
        robot_cfg = self.config.get('robot', {})
        self.wheel_radius = robot_cfg.get('wheel_radius', 0.05)
        self.wheel_base = robot_cfg.get('wheel_base', 0.3)
        self.max_speed = robot_cfg.get('max_speed', 1.0)
        self.camera_height = robot_cfg.get('camera_height', 0.3)
        self.camera_baseline = robot_cfg.get('camera_baseline', 0.1)
        
        # Robot body ID (set when spawned)
        self.robot_id = None
        
        # Body dimensions
        self.body_radius = 0.2
        self.body_height = 0.15
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {'robot': {}}
    
    def spawn(self, position: Tuple[float, float] = (0, 0), 
              orientation: float = 0) -> int:
        """
        Spawn the robot in the simulation.
        
        Args:
            position: (x, y) spawn position
            orientation: Initial heading angle in radians
            
        Returns:
            Robot body ID
        """
        # Create robot body (cylinder)
        body_collision = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=self.body_radius,
            height=self.body_height
        )
        body_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=self.body_radius,
            length=self.body_height,
            rgbaColor=[0.3, 0.3, 0.8, 1]
        )
        
        # Create camera mount (small box on top)
        camera_mount_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.05, self.camera_baseline/2 + 0.02, 0.03]
        )
        camera_mount_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.05, self.camera_baseline/2 + 0.02, 0.03],
            rgbaColor=[0.2, 0.2, 0.2, 1]
        )
        
        # Left camera visual
        left_camera_visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.02,
            rgbaColor=[0, 0, 0, 1]
        )
        
        # Right camera visual
        right_camera_visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.02,
            rgbaColor=[0, 0, 0, 1]
        )
        
        # LiDAR mount visual
        lidar_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.04,
            length=0.05,
            rgbaColor=[0.8, 0.1, 0.1, 1]
        )
        
        # Create robot as multi-body
        self.robot_id = p.createMultiBody(
            baseMass=5.0,
            baseCollisionShapeIndex=body_collision,
            baseVisualShapeIndex=body_visual,
            basePosition=[position[0], position[1], self.body_height/2 + 0.01],
            baseOrientation=p.getQuaternionFromEuler([0, 0, orientation]),
            linkMasses=[0.1, 0.01, 0.01, 0.05],
            linkCollisionShapeIndices=[camera_mount_collision, -1, -1, -1],
            linkVisualShapeIndices=[camera_mount_visual, left_camera_visual, 
                                    right_camera_visual, lidar_visual],
            linkPositions=[
                [0, 0, self.body_height/2 + self.camera_height],  # Camera mount
                [0.05, -self.camera_baseline/2, self.body_height/2 + self.camera_height],  # Left cam
                [0.05, self.camera_baseline/2, self.body_height/2 + self.camera_height],  # Right cam
                [0, 0, self.body_height/2 + 0.05]  # LiDAR
            ],
            linkOrientations=[
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 1]
            ],
            linkInertialFramePositions=[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            linkInertialFrameOrientations=[[0, 0, 0, 1], [0, 0, 0, 1], 
                                           [0, 0, 0, 1], [0, 0, 0, 1]],
            linkParentIndices=[0, 0, 0, 0],
            linkJointTypes=[p.JOINT_FIXED, p.JOINT_FIXED, p.JOINT_FIXED, p.JOINT_FIXED],
            linkJointAxis=[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        )
        
        # Set friction
        p.changeDynamics(self.robot_id, -1, lateralFriction=0.8)
        
        return self.robot_id
    
    def get_position(self) -> np.ndarray:
        """
        Get current robot position.
        
        Returns:
            Position as [x, y, z]
        """
        if self.robot_id is None:
            return np.zeros(3)
        pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        return np.array(pos)
    
    def get_orientation(self) -> float:
        """
        Get current robot heading angle.
        
        Returns:
            Heading angle in radians
        """
        if self.robot_id is None:
            return 0.0
        _, orn = p.getBasePositionAndOrientation(self.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        return euler[2]  # Yaw angle
    
    def get_pose(self) -> Tuple[np.ndarray, float]:
        """
        Get full pose (position and orientation).
        
        Returns:
            (position, heading_angle)
        """
        return self.get_position(), self.get_orientation()
    
    def get_camera_positions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get positions of left and right cameras.
        
        Returns:
            (left_camera_pos, right_camera_pos, camera_orientation_quaternion)
        """
        pos = self.get_position()
        heading = self.get_orientation()
        
        # Camera height
        camera_z = pos[2] + self.camera_height
        
        # Camera offset in robot frame
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        
        # Forward offset for cameras
        forward_offset = 0.05
        
        # Left camera (negative y in robot frame)
        left_offset = np.array([
            cos_h * forward_offset - sin_h * (-self.camera_baseline/2),
            sin_h * forward_offset + cos_h * (-self.camera_baseline/2),
            0
        ])
        left_pos = pos + left_offset
        left_pos[2] = camera_z
        
        # Right camera (positive y in robot frame)
        right_offset = np.array([
            cos_h * forward_offset - sin_h * (self.camera_baseline/2),
            sin_h * forward_offset + cos_h * (self.camera_baseline/2),
            0
        ])
        right_pos = pos + right_offset
        right_pos[2] = camera_z
        
        # Camera orientation (looking forward)
        camera_orn = p.getQuaternionFromEuler([0, 0, heading])
        
        return left_pos, right_pos, np.array(camera_orn)
    
    def get_lidar_position(self) -> Tuple[np.ndarray, float]:
        """
        Get LiDAR sensor position.
        
        Returns:
            (lidar_position, heading_angle)
        """
        pos = self.get_position()
        heading = self.get_orientation()
        lidar_pos = pos.copy()
        lidar_pos[2] = pos[2] + 0.05
        return lidar_pos, heading
    
    def set_velocity(self, linear: float, angular: float):
        """
        Set robot velocity using differential drive kinematics.
        
        Args:
            linear: Forward velocity (m/s)
            angular: Angular velocity (rad/s)
        """
        if self.robot_id is None:
            return
        
        # Clamp velocities
        linear = np.clip(linear, -self.max_speed, self.max_speed)
        angular = np.clip(angular, -self.max_speed/self.wheel_base, 
                         self.max_speed/self.wheel_base)
        
        # Get current orientation
        heading = self.get_orientation()
        
        # Calculate velocity vector
        vx = linear * np.cos(heading)
        vy = linear * np.sin(heading)
        
        # Apply velocity
        p.resetBaseVelocity(
            self.robot_id,
            linearVelocity=[vx, vy, 0],
            angularVelocity=[0, 0, angular]
        )
    
    def move_to(self, target_pos: Tuple[float, float], speed: float = 0.5) -> bool:
        """
        Simple move-to controller (returns True if reached).
        
        Args:
            target_pos: Target (x, y) position
            speed: Movement speed
            
        Returns:
            True if target reached
        """
        pos = self.get_position()
        heading = self.get_orientation()
        
        # Calculate direction to target
        dx = target_pos[0] - pos[0]
        dy = target_pos[1] - pos[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        if distance < 0.1:
            self.set_velocity(0, 0)
            return True
        
        # Calculate target heading
        target_heading = np.arctan2(dy, dx)
        
        # Calculate heading error
        heading_error = target_heading - heading
        
        # Normalize to [-pi, pi]
        while heading_error > np.pi:
            heading_error -= 2 * np.pi
        while heading_error < -np.pi:
            heading_error += 2 * np.pi
        
        # Simple proportional control
        angular = 2.0 * heading_error
        
        # Reduce speed when turning
        linear = speed * (1 - abs(heading_error) / np.pi)
        linear = max(0.1, linear)
        
        self.set_velocity(linear, angular)
        return False
    
    def check_collision(self) -> bool:
        """
        Check if robot is in collision with any object.
        
        Returns:
            True if collision detected
        """
        if self.robot_id is None:
            return False
        
        contact_points = p.getContactPoints(self.robot_id)
        # Filter out ground contact (we only care about obstacles)
        for contact in contact_points:
            other_body = contact[2]  # bodyB
            if other_body != 0:  # 0 is typically the ground plane
                return True
        return False
    
    def reset(self, position: Tuple[float, float] = (0, 0), 
              orientation: float = 0):
        """
        Reset robot to new pose.
        
        Args:
            position: (x, y) position
            orientation: Heading angle in radians
        """
        if self.robot_id is not None:
            p.resetBasePositionAndOrientation(
                self.robot_id,
                [position[0], position[1], self.body_height/2 + 0.01],
                p.getQuaternionFromEuler([0, 0, orientation])
            )
            p.resetBaseVelocity(self.robot_id, [0, 0, 0], [0, 0, 0])
    
    def remove(self):
        """Remove robot from simulation."""
        if self.robot_id is not None:
            p.removeBody(self.robot_id)
            self.robot_id = None
