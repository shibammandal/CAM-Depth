"""
Live visualization for simulation and training.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
from typing import Optional, Tuple, Callable
import time


class LiveDisplay:
    """
    Real-time visualization of robot navigation and depth estimation.
    """
    
    def __init__(self, 
                 figsize: Tuple[int, int] = (14, 8),
                 update_interval: int = 100):
        """
        Initialize live display.
        
        Args:
            figsize: Figure size (width, height)
            update_interval: Update interval in milliseconds
        """
        self.figsize = figsize
        self.update_interval = update_interval
        
        # Figure and axes
        self.fig = None
        self.axes = {}
        self.plots = {}
        
        # Data buffers
        self.depth_errors = []
        self.rewards = []
        self.step_count = 0
        
        # Animation
        self.animation = None
        self._is_running = False
        
    def setup(self):
        """Set up the display layout."""
        plt.ion()
        self.fig = plt.figure(figsize=self.figsize)
        gs = GridSpec(2, 4, figure=self.fig)
        
        # Left camera view
        self.axes['left_cam'] = self.fig.add_subplot(gs[0, 0])
        self.axes['left_cam'].set_title('Left Camera')
        self.axes['left_cam'].axis('off')
        
        # Right camera view
        self.axes['right_cam'] = self.fig.add_subplot(gs[0, 1])
        self.axes['right_cam'].set_title('Right Camera')
        self.axes['right_cam'].axis('off')
        
        # Depth map
        self.axes['depth'] = self.fig.add_subplot(gs[0, 2])
        self.axes['depth'].set_title('Estimated Depth')
        self.axes['depth'].axis('off')
        
        # Ground truth depth
        self.axes['gt_depth'] = self.fig.add_subplot(gs[0, 3])
        self.axes['gt_depth'].set_title('Ground Truth Depth')
        self.axes['gt_depth'].axis('off')
        
        # LiDAR scan / world view
        self.axes['world'] = self.fig.add_subplot(gs[1, 0])
        self.axes['world'].set_title('Robot & LiDAR')
        self.axes['world'].set_aspect('equal')
        
        # Depth error over time
        self.axes['error'] = self.fig.add_subplot(gs[1, 1])
        self.axes['error'].set_title('Depth Error (MAE)')
        self.axes['error'].set_xlabel('Step')
        self.axes['error'].set_ylabel('Error (m)')
        
        # Reward over time
        self.axes['reward'] = self.fig.add_subplot(gs[1, 2])
        self.axes['reward'].set_title('Reward')
        self.axes['reward'].set_xlabel('Step')
        self.axes['reward'].set_ylabel('Reward')
        
        # Parameters display
        self.axes['params'] = self.fig.add_subplot(gs[1, 3])
        self.axes['params'].set_title('Stereo Parameters')
        self.axes['params'].axis('off')
        
        # Initialize plot objects
        self.plots['left_img'] = self.axes['left_cam'].imshow(np.zeros((240, 320, 3), dtype=np.uint8))
        self.plots['right_img'] = self.axes['right_cam'].imshow(np.zeros((240, 320, 3), dtype=np.uint8))
        self.plots['depth_img'] = self.axes['depth'].imshow(np.zeros((240, 320)), cmap='viridis', vmin=0, vmax=10)
        self.plots['gt_img'] = self.axes['gt_depth'].imshow(np.zeros((240, 320)), cmap='viridis', vmin=0, vmax=10)
        
        self.plots['error_line'], = self.axes['error'].plot([], [], 'b-', linewidth=1)
        self.plots['reward_line'], = self.axes['reward'].plot([], [], 'g-', linewidth=1)
        
        plt.tight_layout()
        plt.show(block=False)
        
    def update(self, 
               left_image: Optional[np.ndarray] = None,
               right_image: Optional[np.ndarray] = None,
               depth_map: Optional[np.ndarray] = None,
               gt_depth: Optional[np.ndarray] = None,
               robot_pos: Optional[np.ndarray] = None,
               goal_pos: Optional[np.ndarray] = None,
               lidar_points: Optional[np.ndarray] = None,
               depth_error: Optional[float] = None,
               reward: Optional[float] = None,
               params: Optional[dict] = None):
        """
        Update display with new data.
        
        Args:
            left_image: Left camera RGB image
            right_image: Right camera RGB image
            depth_map: Estimated depth map
            gt_depth: Ground truth depth map
            robot_pos: Robot position [x, y]
            goal_pos: Goal position [x, y]
            lidar_points: LiDAR point cloud (Nx2)
            depth_error: Current depth MAE
            reward: Current reward
            params: Current stereo parameters
        """
        if self.fig is None:
            self.setup()
        
        # Update images
        if left_image is not None:
            self.plots['left_img'].set_data(left_image)
        
        if right_image is not None:
            self.plots['right_img'].set_data(right_image)
        
        if depth_map is not None:
            # Clip for visualization
            depth_vis = np.clip(depth_map, 0, 10)
            self.plots['depth_img'].set_data(depth_vis)
        
        if gt_depth is not None:
            gt_vis = np.clip(gt_depth, 0, 10)
            self.plots['gt_img'].set_data(gt_vis)
        
        # Update world view
        if robot_pos is not None or goal_pos is not None or lidar_points is not None:
            self.axes['world'].clear()
            self.axes['world'].set_title('Robot & LiDAR')
            self.axes['world'].set_xlim(-12, 12)
            self.axes['world'].set_ylim(-12, 12)
            self.axes['world'].set_aspect('equal')
            self.axes['world'].grid(True, alpha=0.3)
            
            if lidar_points is not None and len(lidar_points) > 0:
                self.axes['world'].scatter(lidar_points[:, 0], lidar_points[:, 1], 
                                          c='red', s=2, alpha=0.5, label='LiDAR')
            
            if robot_pos is not None:
                self.axes['world'].plot(robot_pos[0], robot_pos[1], 'bo', 
                                       markersize=10, label='Robot')
            
            if goal_pos is not None:
                self.axes['world'].plot(goal_pos[0], goal_pos[1], 'g*',
                                       markersize=15, label='Goal')
            
            self.axes['world'].legend(loc='upper right')
        
        # Update error plot
        if depth_error is not None:
            self.depth_errors.append(depth_error)
            x_data = list(range(len(self.depth_errors)))
            self.plots['error_line'].set_data(x_data, self.depth_errors)
            self.axes['error'].relim()
            self.axes['error'].autoscale_view()
        
        # Update reward plot
        if reward is not None:
            self.rewards.append(reward)
            x_data = list(range(len(self.rewards)))
            self.plots['reward_line'].set_data(x_data, self.rewards)
            self.axes['reward'].relim()
            self.axes['reward'].autoscale_view()
        
        # Update parameters display
        if params is not None:
            self.axes['params'].clear()
            self.axes['params'].set_title('Stereo Parameters')
            self.axes['params'].axis('off')
            
            param_text = '\n'.join([f'{k}: {v}' for k, v in params.items()])
            self.axes['params'].text(0.1, 0.5, param_text, 
                                    transform=self.axes['params'].transAxes,
                                    fontsize=10, verticalalignment='center',
                                    fontfamily='monospace')
        
        self.step_count += 1
        
        # Redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    def reset(self):
        """Reset display for new episode."""
        self.depth_errors = []
        self.rewards = []
        self.step_count = 0
        
    def close(self):
        """Close the display."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
