"""
Run LiDAR navigation baseline.
Simple script to demonstrate LiDAR-based navigation.
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
from simulation.sensors import LidarSensor
from navigation.lidar_nav import LidarNavigator
from visualization.live_display import LiveDisplay


def parse_args():
    parser = argparse.ArgumentParser(description='Run LiDAR navigation demo')
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
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("CAM-Depth: LiDAR Navigation Demo")
    print("=" * 60)
    
    # Initialize simulation
    print("\nInitializing simulation...")
    world = SimulationWorld(args.config, gui=not args.headless)
    world.initialize()
    world.create_walls()
    
    robot = DifferentialDriveRobot(args.config)
    lidar = LidarSensor(args.config)
    navigator = LidarNavigator()
    
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
        
        # Navigation loop
        for step in range(args.max_steps):
            # Get robot state
            pos = robot.get_position()
            heading = robot.get_orientation()
            
            # LiDAR scan
            lidar_pos, lidar_heading = robot.get_lidar_position()
            distances, angles = lidar.scan(lidar_pos, lidar_heading)
            point_cloud = lidar.get_point_cloud(lidar_pos, lidar_heading)
            
            # Check goal
            dist_to_goal = np.linalg.norm(pos[:2] - goal_pos)
            if dist_to_goal < 0.5:
                print(f"✓ Goal reached in {step} steps!")
                break
            
            # Compute velocity
            linear, angular = navigator.compute_velocity(
                pos, heading, goal_pos, distances, angles
            )
            
            # Apply velocity
            robot.set_velocity(linear, angular)
            
            # Step simulation
            for _ in range(10):
                world.step()
            
            # Update display
            if display and step % 10 == 0:
                display.update(
                    robot_pos=pos,
                    goal_pos=goal_pos,
                    lidar_points=point_cloud
                )
            
            # Small delay for visualization
            if not args.headless:
                time.sleep(0.02)
        
        else:
            print(f"✗ Did not reach goal within {args.max_steps} steps")
        
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
