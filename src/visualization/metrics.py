"""
Metrics visualization and comparison plotting.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import os


class MetricsPlotter:
    """
    Generate plots for training metrics and LiDAR vs Camera comparison.
    """
    
    def __init__(self, save_dir: str = "data/results"):
        """
        Initialize metrics plotter.
        
        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def plot_training_curves(self,
                             depth_errors: List[float],
                             rewards: List[float],
                             epochs: Optional[List[int]] = None,
                             save_name: str = "training_curves.png"):
        """
        Plot training curves.
        
        Args:
            depth_errors: List of depth MAE values
            rewards: List of reward values
            epochs: Optional epoch markers
            save_name: Filename to save
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Depth error
        axes[0].plot(depth_errors, 'b-', alpha=0.6, linewidth=0.5)
        # Moving average
        if len(depth_errors) > 50:
            window = 50
            ma = np.convolve(depth_errors, np.ones(window)/window, mode='valid')
            axes[0].plot(range(window-1, len(depth_errors)), ma, 'b-', linewidth=2, label='Moving Avg')
        axes[0].set_xlabel('Step')
        axes[0].set_ylabel('Depth Error (MAE, meters)')
        axes[0].set_title('Depth Estimation Error Over Training')
        axes[0].grid(True, alpha=0.3)
        
        # Add epoch markers
        if epochs:
            for e in epochs:
                axes[0].axvline(x=e, color='gray', linestyle='--', alpha=0.5)
        
        # Rewards
        axes[1].plot(rewards, 'g-', alpha=0.6, linewidth=0.5)
        if len(rewards) > 50:
            ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
            axes[1].plot(range(window-1, len(rewards)), ma, 'g-', linewidth=2, label='Moving Avg')
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Reward')
        axes[1].set_title('Reward Over Training')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=150)
        plt.close()
        
    def plot_comparison(self,
                        lidar_metrics: Dict[str, List[float]],
                        camera_metrics: Dict[str, List[float]],
                        save_name: str = "comparison.png"):
        """
        Plot LiDAR vs Camera navigation comparison.
        
        Args:
            lidar_metrics: Metrics from LiDAR navigation
            camera_metrics: Metrics from camera navigation
            save_name: Filename to save
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Success rate
        ax = axes[0, 0]
        lidar_success = np.mean(lidar_metrics.get('goal_reached', [0]))
        camera_success = np.mean(camera_metrics.get('goal_reached', [0]))
        bars = ax.bar(['LiDAR', 'Camera'], [lidar_success * 100, camera_success * 100],
                     color=['blue', 'orange'])
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Navigation Success Rate')
        ax.set_ylim(0, 100)
        for bar, val in zip(bars, [lidar_success, camera_success]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                   f'{val*100:.1f}%', ha='center', fontsize=12)
        
        # Path efficiency
        ax = axes[0, 1]
        lidar_eff = lidar_metrics.get('efficiency', [0])
        camera_eff = camera_metrics.get('efficiency', [0])
        ax.boxplot([lidar_eff, camera_eff], labels=['LiDAR', 'Camera'])
        ax.set_ylabel('Efficiency (optimal/actual path)')
        ax.set_title('Path Efficiency')
        
        # Navigation time
        ax = axes[1, 0]
        lidar_time = lidar_metrics.get('elapsed_time', [0])
        camera_time = camera_metrics.get('elapsed_time', [0])
        ax.boxplot([lidar_time, camera_time], labels=['LiDAR', 'Camera'])
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Navigation Time')
        
        # Collisions
        ax = axes[1, 1]
        lidar_coll = np.mean(lidar_metrics.get('num_collisions', [0]))
        camera_coll = np.mean(camera_metrics.get('num_collisions', [0]))
        bars = ax.bar(['LiDAR', 'Camera'], [lidar_coll, camera_coll],
                     color=['blue', 'orange'])
        ax.set_ylabel('Average Collisions per Episode')
        ax.set_title('Collision Rate')
        for bar, val in zip(bars, [lidar_coll, camera_coll]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{val:.2f}', ha='center', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=150)
        plt.close()
        
    def plot_depth_quality_evolution(self,
                                     epochs: List[int],
                                     mae_values: List[float],
                                     accuracy_values: List[float],
                                     save_name: str = "depth_evolution.png"):
        """
        Plot depth estimation quality over training epochs.
        
        Args:
            epochs: Epoch numbers
            mae_values: Mean absolute error at each epoch
            accuracy_values: Accuracy (% within 10%) at each epoch
            save_name: Filename to save
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # MAE
        axes[0].plot(epochs, mae_values, 'b-o', linewidth=2, markersize=4)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Mean Absolute Error (m)')
        axes[0].set_title('Depth Estimation Error vs Training')
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(epochs, [a * 100 for a in accuracy_values], 'g-o', linewidth=2, markersize=4)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Depth Accuracy (within 10% of ground truth)')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=150)
        plt.close()
        
    def plot_cost_analysis(self,
                           lidar_success: float,
                           camera_success: float,
                           lidar_cost: float = 1000.0,
                           camera_cost: float = 100.0,
                           save_name: str = "cost_analysis.png"):
        """
        Plot cost-effectiveness analysis.
        
        Args:
            lidar_success: LiDAR success rate
            camera_success: Camera success rate
            lidar_cost: Simulated LiDAR cost
            camera_cost: Simulated camera cost (per camera, need 2)
            save_name: Filename to save
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Hardware cost comparison
        ax = axes[0]
        stereo_cost = camera_cost * 2
        bars = ax.bar(['LiDAR', 'Stereo Cameras'], [lidar_cost, stereo_cost],
                     color=['blue', 'orange'])
        ax.set_ylabel('Cost ($)')
        ax.set_title('Hardware Cost Comparison')
        for bar, val in zip(bars, [lidar_cost, stereo_cost]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                   f'${val:.0f}', ha='center', fontsize=12)
        
        # Cost-effectiveness (success per $100)
        ax = axes[1]
        lidar_ce = (lidar_success * 100) / (lidar_cost / 100)
        camera_ce = (camera_success * 100) / (stereo_cost / 100)
        bars = ax.bar(['LiDAR', 'Stereo Cameras'], [lidar_ce, camera_ce],
                     color=['blue', 'orange'])
        ax.set_ylabel('Success Rate per $100')
        ax.set_title('Cost Effectiveness')
        for bar, val in zip(bars, [lidar_ce, camera_ce]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{val:.1f}%', ha='center', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=150)
        plt.close()
        
    def plot_parameter_evolution(self,
                                 param_history: Dict[str, List[float]],
                                 save_name: str = "param_evolution.png"):
        """
        Plot how stereo parameters evolved during training.
        
        Args:
            param_history: Dictionary mapping parameter names to value histories
            save_name: Filename to save
        """
        n_params = len(param_history)
        n_cols = 3
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
        axes = np.atleast_2d(axes).flatten()
        
        for ax, (param_name, values) in zip(axes, param_history.items()):
            ax.plot(values, 'b-', linewidth=1)
            ax.set_xlabel('Step')
            ax.set_ylabel('Value')
            ax.set_title(param_name)
            ax.grid(True, alpha=0.3)
        
        # Hide unused axes
        for ax in axes[n_params:]:
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=150)
        plt.close()
