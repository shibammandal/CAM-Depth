"""
Training script for RL-based depth parameter optimization.
"""

import numpy as np
from typing import Dict, Optional, Callable
import os
import yaml
from datetime import datetime

try:
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("Warning: stable-baselines3 not installed. RL training disabled.")


class DepthOptimizationCallback(BaseCallback):
    """
    Custom callback for logging depth optimization progress.
    """
    
    def __init__(self, 
                 log_freq: int = 100,
                 save_path: Optional[str] = None,
                 verbose: int = 1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.save_path = save_path
        
        # Tracking
        self.episode_rewards = []
        self.episode_depths_errors = []
        self.best_mean_reward = -float('inf')
        self.epoch = 0
        
    def _on_step(self) -> bool:
        # Log at intervals
        if self.n_calls % self.log_freq == 0:
            # Get info from environment
            infos = self.locals.get('infos', [{}])
            
            for info in infos:
                if 'metrics' in info:
                    mae = info['metrics'].get('mae', float('inf'))
                    self.episode_depths_errors.append(mae)
                
                if 'episode_reward' in info:
                    self.episode_rewards.append(info['episode_reward'])
            
            if self.verbose > 0 and len(self.episode_depths_errors) > 0:
                mean_mae = np.mean(self.episode_depths_errors[-10:])
                mean_reward = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0
                
                print(f"Step {self.n_calls} | "
                      f"Mean Depth Error: {mean_mae:.4f}m | "
                      f"Mean Reward: {mean_reward:.2f}")
        
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at end of each rollout."""
        self.epoch += 1
        
        if len(self.episode_rewards) > 0:
            mean_reward = np.mean(self.episode_rewards[-10:])
            
            # Save best model
            if mean_reward > self.best_mean_reward and self.save_path:
                self.best_mean_reward = mean_reward
                self.model.save(os.path.join(self.save_path, 'best_model'))
                
                if self.verbose > 0:
                    print(f"New best model saved! Mean reward: {mean_reward:.2f}")


def train_depth_agent(
    env,
    algorithm: str = 'PPO',
    total_timesteps: int = 100000,
    learning_rate: float = 3e-4,
    save_path: str = 'data/results',
    verbose: int = 1,
    callback: Optional[Callable] = None
) -> Dict:
    """
    Train an RL agent to optimize depth estimation parameters.
    
    Args:
        env: Gymnasium environment
        algorithm: RL algorithm ('PPO' or 'SAC')
        total_timesteps: Total training timesteps
        learning_rate: Learning rate
        save_path: Path to save models and logs
        verbose: Verbosity level
        callback: Optional additional callback
        
    Returns:
        Dictionary with training results
    """
    if not HAS_SB3:
        raise ImportError("stable-baselines3 required for training")
    
    # Create save directory
    os.makedirs(save_path, exist_ok=True)
    
    # Wrap environment
    vec_env = DummyVecEnv([lambda: env])
    
    # Create model
    if algorithm.upper() == 'PPO':
        model = PPO(
            'MlpPolicy',
            vec_env,
            learning_rate=learning_rate,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            verbose=verbose,
            tensorboard_log=os.path.join(save_path, 'tensorboard')
        )
    elif algorithm.upper() == 'SAC':
        model = SAC(
            'MlpPolicy',
            vec_env,
            learning_rate=learning_rate,
            buffer_size=100000,
            batch_size=256,
            verbose=verbose,
            tensorboard_log=os.path.join(save_path, 'tensorboard')
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Create callback
    depth_callback = DepthOptimizationCallback(
        log_freq=100,
        save_path=save_path,
        verbose=verbose
    )
    
    callbacks = [depth_callback]
    if callback:
        callbacks.append(callback)
    
    # Train
    start_time = datetime.now()
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True
    )
    end_time = datetime.now()
    
    # Save final model
    model.save(os.path.join(save_path, 'final_model'))
    
    # Get best parameters from callback
    results = {
        'training_time': (end_time - start_time).total_seconds(),
        'total_timesteps': total_timesteps,
        'best_mean_reward': depth_callback.best_mean_reward,
        'final_depth_errors': depth_callback.episode_depths_errors[-100:] if depth_callback.episode_depths_errors else [],
        'algorithm': algorithm,
    }
    
    # Save results
    results_file = os.path.join(save_path, 'training_results.yaml')
    with open(results_file, 'w') as f:
        yaml.dump(results, f)
    
    return results


def evaluate_agent(model, env, num_episodes: int = 10) -> Dict:
    """
    Evaluate a trained agent.
    
    Args:
        model: Trained RL model
        env: Environment
        num_episodes: Number of evaluation episodes
        
    Returns:
        Evaluation metrics
    """
    episode_rewards = []
    depth_errors = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            
            if 'metrics' in info:
                depth_errors.append(info['metrics'].get('mae', 0))
        
        episode_rewards.append(episode_reward)
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_depth_error': np.mean(depth_errors),
        'std_depth_error': np.std(depth_errors),
        'best_params': info.get('best_params', {}),
    }
