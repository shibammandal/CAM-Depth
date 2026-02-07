"""
Main training script for RL-based depth parameter optimization.
Run this to train the agent to optimize stereo matching parameters.
"""

import argparse
import os
import sys
import time
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from rl.env import DepthOptimizationEnv
from rl.training import train_depth_agent, evaluate_agent
from visualization.live_display import LiveDisplay
from visualization.metrics import MetricsPlotter


def parse_args():
    parser = argparse.ArgumentParser(description='Train depth optimization agent')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--timesteps', type=int, default=50000,
                       help='Total training timesteps (default: 50000)')
    parser.add_argument('--algorithm', type=str, default='PPO',
                       choices=['PPO', 'SAC'],
                       help='RL algorithm (default: PPO)')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate (default: 0.0003)')
    parser.add_argument('--visualize', action='store_true',
                       help='Show live visualization during training')
    parser.add_argument('--headless', action='store_true',
                       help='Run without PyBullet GUI')
    parser.add_argument('--test-mode', action='store_true',
                       help='Quick test with minimal timesteps')
    parser.add_argument('--save-dir', type=str, default='data/results',
                       help='Directory to save results')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("CAM-Depth: RL-based Stereo Depth Parameter Optimization")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Algorithm: {args.algorithm}")
    print(f"  Timesteps: {args.timesteps if not args.test_mode else 1000}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Save Directory: {args.save_dir}")
    print(f"  Headless Mode: {args.headless}")
    print(f"  Visualization: {args.visualize}")
    print()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Create environment
    print("[1/4] Creating environment...")
    env = DepthOptimizationEnv(
        config_path=args.config,
        render_mode='human' if args.visualize else None,
        headless=args.headless or args.test_mode
    )
    
    # Set up visualization if enabled
    display = None
    if args.visualize:
        print("[2/4] Setting up visualization...")
        display = LiveDisplay()
        display.setup()
    else:
        print("[2/4] Skipping visualization (use --visualize to enable)")
    
    # Training
    print("[3/4] Starting training...")
    print("-" * 40)
    
    timesteps = 1000 if args.test_mode else args.timesteps
    
    try:
        results = train_depth_agent(
            env=env,
            algorithm=args.algorithm,
            total_timesteps=timesteps,
            learning_rate=args.lr,
            save_path=args.save_dir,
            verbose=1
        )
        
        print("-" * 40)
        print("\nTraining Complete!")
        print(f"  Training Time: {results['training_time']:.1f} seconds")
        print(f"  Best Mean Reward: {results['best_mean_reward']:.2f}")
        
        if results['final_depth_errors']:
            final_mae = np.mean(results['final_depth_errors'])
            print(f"  Final Depth MAE: {final_mae:.4f} meters")
        
    except Exception as e:
        print(f"\nTraining error: {e}")
        import traceback
        traceback.print_exc()
        results = {}
    
    # Plot results
    print("\n[4/4] Generating plots...")
    plotter = MetricsPlotter(save_dir=args.save_dir)
    
    if 'final_depth_errors' in results and results['final_depth_errors']:
        plotter.plot_training_curves(
            depth_errors=results['final_depth_errors'],
            rewards=[],  # Would need to track this
            save_name='training_curves.png'
        )
        print(f"  Saved: {os.path.join(args.save_dir, 'training_curves.png')}")
    
    # Cleanup
    env.close()
    if display:
        display.close()
    
    print("\n" + "=" * 60)
    print("Training complete! Check results in:", args.save_dir)
    print("=" * 60)


if __name__ == '__main__':
    main()
