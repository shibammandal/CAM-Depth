"""
Reward functions for RL-based depth parameter optimization.
"""

import numpy as np
from typing import Dict, Tuple


def depth_error_reward(estimated_depth: np.ndarray, 
                       ground_truth: np.ndarray,
                       max_depth: float = 10.0) -> Tuple[float, Dict]:
    """
    Calculate reward based on depth estimation error.
    
    Args:
        estimated_depth: Depth map from stereo matching
        ground_truth: Ground truth depth from simulator
        max_depth: Maximum valid depth for comparison
        
    Returns:
        (reward, metrics_dict)
    """
    # Create valid mask
    valid_mask = (
        (estimated_depth > 0.1) & 
        (estimated_depth < max_depth) &
        (ground_truth > 0.1) & 
        (ground_truth < max_depth)
    )
    
    if not np.any(valid_mask):
        return -10.0, {'valid_ratio': 0.0, 'mae': float('inf')}
    
    # Calculate errors
    abs_error = np.abs(estimated_depth[valid_mask] - ground_truth[valid_mask])
    mae = np.mean(abs_error)
    
    # Valid pixel ratio
    valid_ratio = np.sum(valid_mask) / valid_mask.size
    
    # Reward components
    # 1. Negative MAE (want to minimize error)
    error_reward = -mae
    
    # 2. Bonus for high valid pixel ratio
    coverage_reward = valid_ratio * 2.0
    
    # 3. Accuracy bonus (percentage within 10% of ground truth)
    rel_error = abs_error / ground_truth[valid_mask]
    accuracy = np.mean(rel_error < 0.1)
    accuracy_reward = accuracy * 5.0
    
    total_reward = error_reward + coverage_reward + accuracy_reward
    
    metrics = {
        'mae': mae,
        'valid_ratio': valid_ratio,
        'accuracy_10pct': accuracy,
        'error_reward': error_reward,
        'coverage_reward': coverage_reward,
        'accuracy_reward': accuracy_reward,
    }
    
    return total_reward, metrics


def smoothness_reward(depth_map: np.ndarray) -> float:
    """
    Reward for smooth depth maps (penalize noise).
    
    Args:
        depth_map: Estimated depth map
        
    Returns:
        Smoothness reward (higher is better)
    """
    # Calculate gradient magnitude
    grad_x = np.diff(depth_map, axis=1)
    grad_y = np.diff(depth_map, axis=0)
    
    # Penalize high gradients at valid depth regions
    valid_x = (depth_map[:, :-1] < 50) & (depth_map[:, 1:] < 50)
    valid_y = (depth_map[:-1, :] < 50) & (depth_map[1:, :] < 50)
    
    if not np.any(valid_x) or not np.any(valid_y):
        return 0.0
    
    # Calculate smoothness (inverse of gradient magnitude)
    grad_mag = np.sqrt(
        np.mean(grad_x[valid_x] ** 2) + 
        np.mean(grad_y[valid_y] ** 2)
    )
    
    # Convert to reward (smaller gradient = higher reward)
    smoothness = 1.0 / (1.0 + grad_mag)
    
    return smoothness


def navigation_reward(robot_pos: np.ndarray,
                      goal_pos: np.ndarray,
                      prev_distance: float,
                      collision: bool,
                      goal_reached: bool) -> Tuple[float, Dict]:
    """
    Calculate navigation-based reward.
    
    Args:
        robot_pos: Current robot position [x, y]
        goal_pos: Goal position [x, y]
        prev_distance: Previous distance to goal
        collision: Whether collision occurred
        goal_reached: Whether goal was reached
        
    Returns:
        (reward, metrics_dict)
    """
    current_distance = np.linalg.norm(robot_pos[:2] - goal_pos[:2])
    
    # Progress reward
    progress = prev_distance - current_distance
    progress_reward = progress * 10.0  # Scale factor
    
    # Collision penalty
    collision_penalty = -100.0 if collision else 0.0
    
    # Goal bonus
    goal_bonus = 500.0 if goal_reached else 0.0
    
    # Small time penalty to encourage faster navigation
    time_penalty = -0.1
    
    total_reward = progress_reward + collision_penalty + goal_bonus + time_penalty
    
    metrics = {
        'distance_to_goal': current_distance,
        'progress': progress,
        'collision': collision,
        'goal_reached': goal_reached,
    }
    
    return total_reward, metrics


def composite_reward(depth_metrics: Dict,
                     nav_metrics: Dict,
                     weights: Dict = None) -> float:
    """
    Combine depth and navigation rewards.
    
    Args:
        depth_metrics: Metrics from depth estimation
        nav_metrics: Metrics from navigation
        weights: Optional weight dictionary
        
    Returns:
        Combined reward
    """
    if weights is None:
        weights = {
            'depth_error': 1.0,
            'smoothness': 0.1,
            'navigation': 0.5,
        }
    
    # Calculate component rewards
    depth_reward = -depth_metrics.get('mae', 10.0) * weights['depth_error']
    smooth_reward = depth_metrics.get('smoothness', 0) * weights['smoothness']
    
    nav_reward = 0.0
    if 'progress' in nav_metrics:
        nav_reward += nav_metrics['progress'] * weights['navigation']
    if nav_metrics.get('collision', False):
        nav_reward -= 100.0
    if nav_metrics.get('goal_reached', False):
        nav_reward += 500.0
    
    return depth_reward + smooth_reward + nav_reward
