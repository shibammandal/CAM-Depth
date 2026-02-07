"""
LiDAR-based navigation for robot.
Provides baseline navigation performance for comparison with camera-based navigation.
"""

import numpy as np
from typing import Tuple, List, Optional
import heapq


class LidarNavigator:
    """
    LiDAR-based navigation using obstacle avoidance and path planning.
    """
    
    def __init__(self, 
                 safety_margin: float = 0.5,
                 max_speed: float = 0.5,
                 goal_threshold: float = 0.5):
        """
        Initialize LiDAR navigator.
        
        Args:
            safety_margin: Minimum distance to obstacles
            max_speed: Maximum robot speed
            goal_threshold: Distance to consider goal reached
        """
        self.safety_margin = safety_margin
        self.max_speed = max_speed
        self.goal_threshold = goal_threshold
        
        # Navigation state
        self.path = []
        self.current_waypoint_idx = 0
        
    def get_obstacle_direction(self, 
                               lidar_distances: np.ndarray,
                               lidar_angles: np.ndarray) -> Tuple[float, float]:
        """
        Calculate repulsive force from nearby obstacles.
        
        Args:
            lidar_distances: Array of distance readings
            lidar_angles: Array of corresponding angles
            
        Returns:
            (repulsion_x, repulsion_y) - direction away from obstacles
        """
        # Find close obstacles
        close_mask = lidar_distances < self.safety_margin * 2
        
        if not np.any(close_mask):
            return 0.0, 0.0
        
        # Calculate repulsion vector (away from obstacles)
        close_distances = lidar_distances[close_mask]
        close_angles = lidar_angles[close_mask]
        
        # Weight inversely by distance (closer = stronger repulsion)
        weights = 1.0 / (close_distances + 0.1)
        weights = weights / np.sum(weights)
        
        # Direction toward obstacles
        obs_x = np.sum(weights * np.cos(close_angles))
        obs_y = np.sum(weights * np.sin(close_angles))
        
        # Return opposite direction (repulsion)
        return -obs_x, -obs_y
    
    def compute_velocity(self,
                         robot_pos: np.ndarray,
                         robot_heading: float,
                         goal_pos: np.ndarray,
                         lidar_distances: np.ndarray,
                         lidar_angles: np.ndarray) -> Tuple[float, float]:
        """
        Compute robot velocity using potential field navigation.
        
        Args:
            robot_pos: Current robot position [x, y]
            robot_heading: Current heading angle
            goal_pos: Goal position [x, y]
            lidar_distances: LiDAR distance readings
            lidar_angles: LiDAR angles (in robot frame)
            
        Returns:
            (linear_velocity, angular_velocity)
        """
        # Goal attraction
        goal_dir = goal_pos[:2] - robot_pos[:2]
        goal_dist = np.linalg.norm(goal_dir)
        
        if goal_dist < self.goal_threshold:
            return 0.0, 0.0  # Goal reached
        
        goal_dir = goal_dir / goal_dist  # Normalize
        
        # Obstacle repulsion (in world frame)
        rep_x, rep_y = self.get_obstacle_direction(
            lidar_distances, 
            lidar_angles + robot_heading  # Convert to world frame
        )
        
        # Combine forces
        attraction_weight = 1.0
        repulsion_weight = 2.0
        
        combined_x = attraction_weight * goal_dir[0] + repulsion_weight * rep_x
        combined_y = attraction_weight * goal_dir[1] + repulsion_weight * rep_y
        
        # Normalize
        combined_mag = np.sqrt(combined_x**2 + combined_y**2)
        if combined_mag > 0:
            combined_x /= combined_mag
            combined_y /= combined_mag
        
        # Calculate target heading
        target_heading = np.arctan2(combined_y, combined_x)
        
        # Heading error
        heading_error = target_heading - robot_heading
        while heading_error > np.pi:
            heading_error -= 2 * np.pi
        while heading_error < -np.pi:
            heading_error += 2 * np.pi
        
        # Angular velocity (proportional control)
        angular_vel = 2.0 * heading_error
        angular_vel = np.clip(angular_vel, -2.0, 2.0)
        
        # Linear velocity (reduce when turning or near obstacles)
        min_dist = np.min(lidar_distances)
        obstacle_factor = np.clip((min_dist - self.safety_margin) / self.safety_margin, 0, 1)
        turn_factor = 1.0 - min(abs(heading_error) / np.pi, 1.0) * 0.5
        
        linear_vel = self.max_speed * obstacle_factor * turn_factor
        
        return linear_vel, angular_vel
    
    def check_collision_risk(self, 
                             lidar_distances: np.ndarray,
                             threshold: float = 0.3) -> bool:
        """
        Check if there's an imminent collision risk.
        
        Args:
            lidar_distances: LiDAR distance readings
            threshold: Minimum safe distance
            
        Returns:
            True if collision is imminent
        """
        # Check front sector (middle third of readings)
        n = len(lidar_distances)
        front_start = n // 3
        front_end = 2 * n // 3
        front_distances = lidar_distances[front_start:front_end]
        
        return np.any(front_distances < threshold)
    
    def get_navigation_metrics(self,
                               robot_pos: np.ndarray,
                               goal_pos: np.ndarray,
                               start_time: float,
                               current_time: float,
                               path_length: float,
                               num_collisions: int) -> dict:
        """
        Calculate navigation performance metrics.
        
        Args:
            robot_pos: Current position
            goal_pos: Goal position
            start_time: Episode start time
            current_time: Current time
            path_length: Total path length traveled
            num_collisions: Number of collisions
            
        Returns:
            Dictionary of metrics
        """
        distance_to_goal = np.linalg.norm(robot_pos[:2] - goal_pos[:2])
        goal_reached = distance_to_goal < self.goal_threshold
        elapsed_time = current_time - start_time
        
        # Efficiency = straight-line distance / actual path length
        straight_dist = np.linalg.norm(goal_pos[:2] - robot_pos[:2])
        efficiency = straight_dist / max(path_length, 0.01) if goal_reached else 0
        
        return {
            'goal_reached': goal_reached,
            'distance_to_goal': distance_to_goal,
            'elapsed_time': elapsed_time,
            'path_length': path_length,
            'num_collisions': num_collisions,
            'efficiency': efficiency,
        }
