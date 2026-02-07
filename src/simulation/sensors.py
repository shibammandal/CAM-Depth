"""
Sensor implementations for LiDAR and stereo cameras in PyBullet.
"""

import pybullet as p
import numpy as np
from typing import Tuple, Optional
import yaml


class LidarSensor:
    """Simulated 2D LiDAR sensor using PyBullet raycasting."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize LiDAR sensor parameters.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        
        lidar_cfg = self.config.get('sensors', {}).get('lidar', {})
        self.num_rays = lidar_cfg.get('num_rays', 360)
        self.max_range = lidar_cfg.get('max_range', 10.0)
        self.fov = lidar_cfg.get('fov', 360)
        
        # Pre-compute ray angles
        self.angles = np.linspace(
            -np.radians(self.fov/2),
            np.radians(self.fov/2),
            self.num_rays
        )
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {'sensors': {'lidar': {}}}
    
    def scan(self, position: np.ndarray, heading: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform a LiDAR scan from given position.
        
        Args:
            position: [x, y, z] sensor position
            heading: Sensor heading angle in radians
            
        Returns:
            (distances, angles) - arrays of distance readings and corresponding angles
        """
        # Calculate ray directions
        ray_angles = self.angles + heading
        
        # Calculate ray endpoints
        ray_starts = np.tile(position, (self.num_rays, 1))
        ray_ends = np.zeros((self.num_rays, 3))
        ray_ends[:, 0] = position[0] + self.max_range * np.cos(ray_angles)
        ray_ends[:, 1] = position[1] + self.max_range * np.sin(ray_angles)
        ray_ends[:, 2] = position[2]
        
        # Batch raycast
        results = p.rayTestBatch(ray_starts.tolist(), ray_ends.tolist())
        
        # Extract distances
        distances = np.array([
            r[2] * self.max_range if r[0] != -1 else self.max_range
            for r in results
        ])
        
        return distances, self.angles
    
    def get_point_cloud(self, position: np.ndarray, heading: float) -> np.ndarray:
        """
        Get LiDAR scan as 2D point cloud.
        
        Args:
            position: Sensor position
            heading: Sensor heading
            
        Returns:
            Point cloud as Nx2 array of [x, y] points in world frame
        """
        distances, angles = self.scan(position, heading)
        
        # Convert to cartesian coordinates
        ray_angles = angles + heading
        points = np.zeros((self.num_rays, 2))
        points[:, 0] = position[0] + distances * np.cos(ray_angles)
        points[:, 1] = position[1] + distances * np.sin(ray_angles)
        
        # Filter out max range readings
        valid_mask = distances < self.max_range - 0.1
        
        return points[valid_mask]


class StereoCamera:
    """Simulated stereo camera pair using PyBullet rendering."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize stereo camera parameters.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        
        camera_cfg = self.config.get('sensors', {}).get('cameras', {})
        self.width = camera_cfg.get('width', 320)
        self.height = camera_cfg.get('height', 240)
        self.fov = camera_cfg.get('fov', 60)
        self.near = camera_cfg.get('near', 0.1)
        self.far = camera_cfg.get('far', 10.0)
        
        robot_cfg = self.config.get('robot', {})
        self.baseline = robot_cfg.get('camera_baseline', 0.1)
        
        # Calculate camera intrinsics
        self.aspect = self.width / self.height
        self.focal_length = self.width / (2 * np.tan(np.radians(self.fov/2)))
        
        # Projection matrix (computed once)
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=self.aspect,
            nearVal=self.near,
            farVal=self.far
        )
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {'sensors': {'cameras': {}}, 'robot': {}}
    
    def _compute_view_matrix(self, position: np.ndarray, 
                             orientation: np.ndarray) -> list:
        """
        Compute view matrix for camera.
        
        Args:
            position: Camera position [x, y, z]
            orientation: Camera orientation quaternion
            
        Returns:
            PyBullet view matrix
        """
        # Get rotation matrix from quaternion
        rot_matrix = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        
        # Camera looks along positive X axis in robot frame
        forward = rot_matrix @ np.array([1, 0, 0])
        up = rot_matrix @ np.array([0, 0, 1])
        
        target = position + forward
        
        return p.computeViewMatrix(
            cameraEyePosition=position.tolist(),
            cameraTargetPosition=target.tolist(),
            cameraUpVector=up.tolist()
        )
    
    def capture_stereo(self, left_pos: np.ndarray, right_pos: np.ndarray,
                       orientation: np.ndarray, 
                       use_opengl: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Capture stereo image pair.
        
        Args:
            left_pos: Left camera position
            right_pos: Right camera position
            orientation: Camera orientation quaternion (same for both)
            use_opengl: Use OpenGL renderer (faster, requires display)
            
        Returns:
            (left_rgb, right_rgb, left_depth, right_depth)
        """
        renderer = p.ER_BULLET_HARDWARE_OPENGL if use_opengl else p.ER_TINY_RENDERER
        
        # Left camera
        left_view = self._compute_view_matrix(left_pos, orientation)
        _, _, left_rgb, left_depth_buffer, _ = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=left_view,
            projectionMatrix=self.projection_matrix,
            renderer=renderer
        )
        
        # Right camera
        right_view = self._compute_view_matrix(right_pos, orientation)
        _, _, right_rgb, right_depth_buffer, _ = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=right_view,
            projectionMatrix=self.projection_matrix,
            renderer=renderer
        )
        
        # Convert to numpy arrays
        left_rgb = np.array(left_rgb, dtype=np.uint8).reshape(self.height, self.width, 4)[:, :, :3]
        right_rgb = np.array(right_rgb, dtype=np.uint8).reshape(self.height, self.width, 4)[:, :, :3]
        
        # Convert depth buffer to actual depth and ensure 2D shape
        left_depth_arr = np.array(left_depth_buffer).reshape(self.height, self.width)
        right_depth_arr = np.array(right_depth_buffer).reshape(self.height, self.width)
        left_depth = self.far * self.near / (self.far - (self.far - self.near) * left_depth_arr)
        right_depth = self.far * self.near / (self.far - (self.far - self.near) * right_depth_arr)
        
        return left_rgb, right_rgb, left_depth, right_depth
    
    def get_ground_truth_depth(self, position: np.ndarray, 
                               orientation: np.ndarray,
                               use_opengl: bool = True) -> np.ndarray:
        """
        Get ground truth depth from a single camera view.
        Used as reference for evaluating stereo depth estimation.
        
        Args:
            position: Camera position
            orientation: Camera orientation quaternion
            use_opengl: Use OpenGL renderer
            
        Returns:
            Ground truth depth map
        """
        renderer = p.ER_BULLET_HARDWARE_OPENGL if use_opengl else p.ER_TINY_RENDERER
        
        view_matrix = self._compute_view_matrix(position, orientation)
        _, _, _, depth_buffer, _ = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=view_matrix,
            projectionMatrix=self.projection_matrix,
            renderer=renderer
        )
        
        # Convert to actual depth values and ensure 2D shape
        depth_arr = np.array(depth_buffer).reshape(self.height, self.width)
        depth = self.far * self.near / (self.far - (self.far - self.near) * depth_arr)
        
        return depth
    
    def get_intrinsics(self) -> dict:
        """
        Get camera intrinsic parameters.
        
        Returns:
            Dictionary with fx, fy, cx, cy
        """
        return {
            'fx': self.focal_length,
            'fy': self.focal_length,
            'cx': self.width / 2,
            'cy': self.height / 2,
            'width': self.width,
            'height': self.height,
            'baseline': self.baseline
        }
