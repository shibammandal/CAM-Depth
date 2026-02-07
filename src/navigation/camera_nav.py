"""
Camera-based navigation using depth maps.
Uses the same navigation algorithm as LiDAR but with depth map input.
"""

import numpy as np
from typing import Tuple, Optional


class CameraNavigator:
    """
    Camera-based navigation using depth maps for obstacle avoidance.
    """
    
    def __init__(self,
                 safety_margin: float = 0.5,
                 max_speed: float = 0.5,
                 goal_threshold: float = 0.5,
                 depth_threshold: float = 2.0):
        """
        Initialize camera navigator.
        
        Args:
            safety_margin: Minimum distance to obstacles
            max_speed: Maximum robot speed
            goal_threshold: Distance to consider goal reached
            depth_threshold: Max depth to consider for obstacles
        """
        self.safety_margin = safety_margin
        self.max_speed = max_speed
        self.goal_threshold = goal_threshold
        self.depth_threshold = depth_threshold
        
    def depth_to_obstacle_vector(self, 
                                  depth_map: np.ndarray,
                                  fov: float = 60.0) -> Tuple[float, float]:
        """
        Convert depth map to obstacle proximity in different directions.
        
        Args:
            depth_map: 2D depth map (HxW)
            fov: Camera field of view in degrees
            
        Returns:
            (obstacle_distance, obstacle_angle) - closest obstacle info
        """
        height, width = depth_map.shape
        
        # Divide image into angular sectors
        num_sectors = 9
        sector_width = width // num_sectors
        
        sector_distances = []
        for i in range(num_sectors):
            start_col = i * sector_width
            end_col = start_col + sector_width
            sector = depth_map[:, start_col:end_col]
            
            # Use median of lower half (ground-level obstacles)
            lower_half = sector[height//2:, :]
            valid_depths = lower_half[(lower_half > 0.1) & (lower_half < self.depth_threshold)]
            
            if len(valid_depths) > 0:
                sector_distances.append(np.median(valid_depths))
            else:
                sector_distances.append(self.depth_threshold)
        
        sector_distances = np.array(sector_distances)
        
        # Calculate angles for each sector
        angles = np.linspace(-np.radians(fov/2), np.radians(fov/2), num_sectors)
        
        # Find minimum distance and its angle
        min_idx = np.argmin(sector_distances)
        min_distance = sector_distances[min_idx]
        obstacle_angle = angles[min_idx]
        
        return min_distance, obstacle_angle
    
    def get_repulsion_vector(self, 
                             depth_map: np.ndarray,
                             fov: float = 60.0) -> Tuple[float, float]:
        """
        Calculate repulsion vector from depth map.
        
        Args:
            depth_map: 2D depth map
            fov: Camera field of view
            
        Returns:
            (repulsion_x, repulsion_y) in camera frame
        """
        height, width = depth_map.shape
        
        # Divide into left, center, right regions
        third = width // 3
        
        left_region = depth_map[:, :third]
        center_region = depth_map[:, third:2*third]
        right_region = depth_map[:, 2*third:]
        
        def get_min_depth(region):
            valid = region[(region > 0.1) & (region < self.depth_threshold)]
            return np.median(valid) if len(valid) > 0 else self.depth_threshold
        
        left_dist = get_min_depth(left_region)
        center_dist = get_min_depth(center_region)
        right_dist = get_min_depth(right_region)
        
        # If all clear, no repulsion
        if min(left_dist, center_dist, right_dist) > self.safety_margin * 2:
            return 0.0, 0.0
        
        # Calculate repulsion (push away from obstacles)
        # Closer obstacles create stronger repulsion
        left_weight = 1.0 / (left_dist + 0.1)
        center_weight = 1.0 / (center_dist + 0.1)
        right_weight = 1.0 / (right_dist + 0.1)
        
        # Left obstacles push right, right obstacles push left
        lateral_repulsion = (left_weight - right_weight) / (left_weight + right_weight + 0.01)
        
        # Front obstacles push backward
        forward_repulsion = -center_weight / (left_weight + center_weight + right_weight + 0.01)
        
        return forward_repulsion, lateral_repulsion
    
    def compute_velocity(self,
                         robot_pos: np.ndarray,
                         robot_heading: float,
                         goal_pos: np.ndarray,
                         depth_map: np.ndarray,
                         fov: float = 60.0) -> Tuple[float, float]:
        """
        Compute robot velocity using depth-based navigation.
        
        Args:
            robot_pos: Current robot position [x, y]
            robot_heading: Current heading angle
            goal_pos: Goal position [x, y]
            depth_map: Depth map from stereo cameras
            fov: Camera field of view
            
        Returns:
            (linear_velocity, angular_velocity)
        """
        # Goal attraction
        goal_dir = goal_pos[:2] - robot_pos[:2]
        goal_dist = np.linalg.norm(goal_dir)
        
        if goal_dist < self.goal_threshold:
            return 0.0, 0.0  # Goal reached
        
        goal_dir = goal_dir / goal_dist
        
        # Target heading toward goal
        target_heading = np.arctan2(goal_dir[1], goal_dir[0])
        
        # Obstacle repulsion from depth map
        fwd_rep, lat_rep = self.get_repulsion_vector(depth_map, fov)
        
        # Get minimum distance for speed control
        min_dist, _ = self.depth_to_obstacle_vector(depth_map, fov)
        
        # Modify target heading based on repulsion
        heading_modifier = lat_rep * 1.0  # Scale factor
        
        # Combine goal heading with obstacle avoidance
        adjusted_heading = target_heading + heading_modifier
        
        # Heading error
        heading_error = adjusted_heading - robot_heading
        while heading_error > np.pi:
            heading_error -= 2 * np.pi
        while heading_error < -np.pi:
            heading_error += 2 * np.pi
        
        # Angular velocity
        angular_vel = 2.0 * heading_error
        angular_vel = np.clip(angular_vel, -2.0, 2.0)
        
        # Linear velocity
        obstacle_factor = np.clip((min_dist - self.safety_margin) / self.safety_margin, 0, 1)
        turn_factor = 1.0 - min(abs(heading_error) / np.pi, 1.0) * 0.5
        
        # Add penalty for front obstacles
        if fwd_rep < -0.3:  # Strong forward repulsion
            obstacle_factor *= 0.5
        
        linear_vel = self.max_speed * obstacle_factor * turn_factor
        
        return linear_vel, angular_vel
    
    def check_collision_risk(self, 
                             depth_map: np.ndarray,
                             threshold: float = 0.3) -> bool:
        """
        Check if there's an imminent collision risk.
        
        Args:
            depth_map: Depth map
            threshold: Minimum safe distance
            
        Returns:
            True if collision is imminent
        """
        height, width = depth_map.shape
        
        # Check center region of lower half
        center_start = width // 3
        center_end = 2 * width // 3
        lower_start = height // 2
        
        center_lower = depth_map[lower_start:, center_start:center_end]
        valid_depths = center_lower[(center_lower > 0.1) & (center_lower < 50)]
        
        if len(valid_depths) == 0:
            return False
        
        min_depth = np.min(valid_depths)
        return min_depth < threshold
    
    def get_navigation_metrics(self,
                               robot_pos: np.ndarray,
                               goal_pos: np.ndarray,
                               start_time: float,
                               current_time: float,
                               path_length: float,
                               num_collisions: int,
                               depth_errors: list) -> dict:
        """
        Calculate navigation and depth performance metrics.
        
        Args:
            robot_pos: Current position
            goal_pos: Goal position
            start_time: Episode start time
            current_time: Current time
            path_length: Total path length
            num_collisions: Number of collisions
            depth_errors: List of depth estimation errors
            
        Returns:
            Dictionary of metrics
        """
        distance_to_goal = np.linalg.norm(robot_pos[:2] - goal_pos[:2])
        goal_reached = distance_to_goal < self.goal_threshold
        elapsed_time = current_time - start_time
        
        # Efficiency
        straight_dist = np.linalg.norm(goal_pos[:2] - robot_pos[:2])
        efficiency = straight_dist / max(path_length, 0.01) if goal_reached else 0
        
        # Depth metrics
        mean_depth_error = np.mean(depth_errors) if depth_errors else 0
        
        return {
            'goal_reached': goal_reached,
            'distance_to_goal': distance_to_goal,
            'elapsed_time': elapsed_time,
            'path_length': path_length,
            'num_collisions': num_collisions,
            'efficiency': efficiency,
            'mean_depth_error': mean_depth_error,
        }
