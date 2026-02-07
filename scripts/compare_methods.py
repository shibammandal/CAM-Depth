"""
Compare LiDAR-based and Camera-based navigation performance.
Runs multiple navigation episodes with each method and generates comparison plots.
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
from simulation.sensors import LidarSensor, StereoCamera
from depth.stereo_matching import StereoDepthEstimator
from navigation.lidar_nav import LidarNavigator
from navigation.camera_nav import CameraNavigator
from visualization.metrics import MetricsPlotter


def parse_args():
    parser = argparse.ArgumentParser(description='Compare LiDAR vs Camera navigation')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes to run (default: 10)')
    parser.add_argument('--max-steps', type=int, default=500,
                       help='Max steps per episode (default: 500)')
    parser.add_argument('--headless', action='store_true',
                       help='Run without GUI')
    parser.add_argument('--save-dir', type=str, default='data/results',
                       help='Directory to save results')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    return parser.parse_args()


def run_lidar_episode(world, robot, lidar, navigator, goal_pos, max_steps):
    """Run one navigation episode using LiDAR."""
    metrics_log = {
        'goal_reached': False,
        'elapsed_time': 0,
        'path_length': 0,
        'num_collisions': 0,
        'efficiency': 0,
    }
    
    start_pos = robot.get_position()[:2].copy()
    prev_pos = start_pos.copy()
    path_length = 0
    num_collisions = 0
    
    start_time = time.time()
    
    for step in range(max_steps):
        # Get robot state
        pos = robot.get_position()
        heading = robot.get_orientation()
        
        # LiDAR scan
        lidar_pos, lidar_heading = robot.get_lidar_position()
        distances, angles = lidar.scan(lidar_pos, lidar_heading)
        
        # Check goal
        dist_to_goal = np.linalg.norm(pos[:2] - goal_pos)
        if dist_to_goal < 0.5:
            metrics_log['goal_reached'] = True
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
        
        # Check collision
        if robot.check_collision():
            num_collisions += 1
        
        # Track path length
        new_pos = robot.get_position()[:2]
        path_length += np.linalg.norm(new_pos - prev_pos)
        prev_pos = new_pos.copy()
    
    elapsed_time = time.time() - start_time
    
    # Calculate efficiency
    straight_dist = np.linalg.norm(goal_pos - start_pos)
    efficiency = straight_dist / max(path_length, 0.01) if metrics_log['goal_reached'] else 0
    
    metrics_log['elapsed_time'] = elapsed_time
    metrics_log['path_length'] = path_length
    metrics_log['num_collisions'] = num_collisions
    metrics_log['efficiency'] = efficiency
    
    return metrics_log


def run_camera_episode(world, robot, stereo_camera, depth_estimator, 
                       navigator, goal_pos, max_steps):
    """Run one navigation episode using stereo cameras."""
    metrics_log = {
        'goal_reached': False,
        'elapsed_time': 0,
        'path_length': 0,
        'num_collisions': 0,
        'efficiency': 0,
        'depth_errors': [],
    }
    
    start_pos = robot.get_position()[:2].copy()
    prev_pos = start_pos.copy()
    path_length = 0
    num_collisions = 0
    depth_errors = []
    
    start_time = time.time()
    
    for step in range(max_steps):
        # Get robot state
        pos = robot.get_position()
        heading = robot.get_orientation()
        
        # Get stereo images
        left_pos, right_pos, orientation = robot.get_camera_positions()
        left_rgb, right_rgb, left_gt, right_gt = stereo_camera.capture_stereo(
            left_pos, right_pos, orientation, use_opengl=True
        )
        
        # Estimate depth
        estimated_depth, metrics = depth_estimator.compute_depth_with_quality(
            left_rgb, right_rgb, left_gt
        )
        if 'mae' in metrics:
            depth_errors.append(metrics['mae'])
        
        # Check goal
        dist_to_goal = np.linalg.norm(pos[:2] - goal_pos)
        if dist_to_goal < 0.5:
            metrics_log['goal_reached'] = True
            break
        
        # Compute velocity using depth
        linear, angular = navigator.compute_velocity(
            pos, heading, goal_pos, estimated_depth, stereo_camera.fov
        )
        
        # Apply velocity
        robot.set_velocity(linear, angular)
        
        # Step simulation
        for _ in range(10):
            world.step()
        
        # Check collision
        if robot.check_collision():
            num_collisions += 1
        
        # Track path length
        new_pos = robot.get_position()[:2]
        path_length += np.linalg.norm(new_pos - prev_pos)
        prev_pos = new_pos.copy()
    
    elapsed_time = time.time() - start_time
    
    # Calculate efficiency
    straight_dist = np.linalg.norm(goal_pos - start_pos)
    efficiency = straight_dist / max(path_length, 0.01) if metrics_log['goal_reached'] else 0
    
    metrics_log['elapsed_time'] = elapsed_time
    metrics_log['path_length'] = path_length
    metrics_log['num_collisions'] = num_collisions
    metrics_log['efficiency'] = efficiency
    metrics_log['depth_errors'] = depth_errors
    
    return metrics_log


def main():
    args = parse_args()
    
    print("=" * 60)
    print("CAM-Depth: LiDAR vs Camera Navigation Comparison")
    print("=" * 60)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Initialize simulation
    print("\n[1/4] Initializing simulation...")
    world = SimulationWorld(args.config, gui=not args.headless)
    world.initialize()
    world.create_walls()
    
    robot = DifferentialDriveRobot(args.config)
    lidar = LidarSensor(args.config)
    stereo_camera = StereoCamera(args.config)
    
    depth_estimator = StereoDepthEstimator(args.config)
    intrinsics = stereo_camera.get_intrinsics()
    depth_estimator.set_camera_params(intrinsics['fx'], intrinsics['baseline'])
    
    lidar_navigator = LidarNavigator()
    camera_navigator = CameraNavigator()
    
    # Run LiDAR episodes
    print("\n[2/4] Running LiDAR navigation episodes...")
    lidar_metrics = {
        'goal_reached': [],
        'elapsed_time': [],
        'path_length': [],
        'num_collisions': [],
        'efficiency': [],
    }
    
    for ep in range(args.episodes):
        # Random start and goal
        start_pos = np.array([np.random.uniform(-7, -3), np.random.uniform(-7, -3)])
        goal_pos = np.array([np.random.uniform(3, 7), np.random.uniform(3, 7)])
        
        # Setup scene
        world.reset()
        world.create_random_obstacles(10, 2.0, tuple(start_pos), tuple(goal_pos))
        world.set_markers(tuple(start_pos), tuple(goal_pos))
        
        robot.spawn(tuple(start_pos), 0)
        
        # Run episode
        metrics = run_lidar_episode(world, robot, lidar, lidar_navigator, 
                                    goal_pos, args.max_steps)
        
        for key in lidar_metrics:
            lidar_metrics[key].append(metrics[key])
        
        status = "✓" if metrics['goal_reached'] else "✗"
        print(f"  Episode {ep+1}/{args.episodes}: {status} "
              f"(time: {metrics['elapsed_time']:.1f}s, collisions: {metrics['num_collisions']})")
        
        robot.remove()
    
    # Run Camera episodes
    print("\n[3/4] Running Camera navigation episodes...")
    camera_metrics = {
        'goal_reached': [],
        'elapsed_time': [],
        'path_length': [],
        'num_collisions': [],
        'efficiency': [],
        'depth_errors': [],
    }
    
    for ep in range(args.episodes):
        # Random start and goal
        start_pos = np.array([np.random.uniform(-7, -3), np.random.uniform(-7, -3)])
        goal_pos = np.array([np.random.uniform(3, 7), np.random.uniform(3, 7)])
        
        # Setup scene
        world.reset()
        world.create_random_obstacles(10, 2.0, tuple(start_pos), tuple(goal_pos))
        world.set_markers(tuple(start_pos), tuple(goal_pos))
        
        robot.spawn(tuple(start_pos), 0)
        
        # Run episode
        metrics = run_camera_episode(world, robot, stereo_camera, depth_estimator,
                                     camera_navigator, goal_pos, args.max_steps)
        
        for key in ['goal_reached', 'elapsed_time', 'path_length', 'num_collisions', 'efficiency']:
            camera_metrics[key].append(metrics[key])
        camera_metrics['depth_errors'].extend(metrics['depth_errors'])
        
        status = "✓" if metrics['goal_reached'] else "✗"
        mean_depth_err = np.mean(metrics['depth_errors']) if metrics['depth_errors'] else 0
        print(f"  Episode {ep+1}/{args.episodes}: {status} "
              f"(time: {metrics['elapsed_time']:.1f}s, depth_err: {mean_depth_err:.3f}m)")
        
        robot.remove()
    
    # Generate plots
    print("\n[4/4] Generating comparison plots...")
    plotter = MetricsPlotter(save_dir=args.save_dir)
    plotter.plot_comparison(lidar_metrics, camera_metrics, 'lidar_vs_camera.png')
    
    # Cost analysis
    lidar_success = np.mean(lidar_metrics['goal_reached'])
    camera_success = np.mean(camera_metrics['goal_reached'])
    plotter.plot_cost_analysis(lidar_success, camera_success, 
                               lidar_cost=1000, camera_cost=100,
                               save_name='cost_analysis.png')
    
    # Print summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"\n{'Metric':<25} {'LiDAR':>15} {'Camera':>15}")
    print("-" * 55)
    print(f"{'Success Rate':<25} {np.mean(lidar_metrics['goal_reached'])*100:>14.1f}% "
          f"{np.mean(camera_metrics['goal_reached'])*100:>14.1f}%")
    print(f"{'Avg Navigation Time':<25} {np.mean(lidar_metrics['elapsed_time']):>14.2f}s "
          f"{np.mean(camera_metrics['elapsed_time']):>14.2f}s")
    print(f"{'Avg Collisions':<25} {np.mean(lidar_metrics['num_collisions']):>15.2f} "
          f"{np.mean(camera_metrics['num_collisions']):>15.2f}")
    print(f"{'Path Efficiency':<25} {np.mean(lidar_metrics['efficiency']):>15.2f} "
          f"{np.mean(camera_metrics['efficiency']):>15.2f}")
    if camera_metrics['depth_errors']:
        print(f"{'Avg Depth Error':<25} {'N/A':>15} "
              f"{np.mean(camera_metrics['depth_errors']):>14.3f}m")
    
    print("\n" + "=" * 60)
    print(f"Results saved to: {args.save_dir}")
    print("=" * 60)
    
    # Cleanup
    world.close()


if __name__ == '__main__':
    main()
