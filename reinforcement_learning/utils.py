import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
import os
from datetime import datetime


def linear_decay_epsilon(episode: int, total_episodes: int, eps_start: float = 1.0, eps_end: float = 0.01) -> float:
    """
    Calculate epsilon value with linear decay.
    
    Args:
        episode: Current episode number
        total_episodes: Total number of episodes
        eps_start: Starting epsilon value
        eps_end: Ending epsilon value
        
    Returns:
        Current epsilon value
    """
    return max(eps_end, eps_start - (episode / total_episodes) * (eps_start - eps_end))


def exponential_decay_epsilon(episode: int, total_episodes: int, eps_start: float = 1.0, eps_end: float = 0.01, decay_rate: float = 0.99) -> float:
    """
    Calculate epsilon value with exponential decay.
    
    Args:
        episode: Current episode number
        total_episodes: Total number of episodes
        eps_start: Starting epsilon value
        eps_end: Ending epsilon value
        decay_rate: Decay rate
        
    Returns:
        Current epsilon value
    """
    return max(eps_end, eps_start * (decay_rate ** episode))


def plot_rewards(rewards: List[float], avg_window: int = 100, save_path: str = None):
    """
    Plot rewards over episodes.
    
    Args:
        rewards: List of rewards per episode
        avg_window: Window size for moving average
        save_path: Path to save the plot (if None, plot is shown)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label='Reward')
    
    # Calculate and plot moving average
    if len(rewards) >= avg_window:
        avg_rewards = np.convolve(rewards, np.ones(avg_window)/avg_window, mode='valid')
        plt.plot(np.arange(avg_window-1, len(rewards)), avg_rewards, label=f'{avg_window}-episode Moving Average')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Rewards Over Episodes')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def plot_metric_comparison(metrics: List[Dict[str, float]], labels: List[str], save_path: str = None):
    """
    Plot comparison of reconstruction quality metrics.
    
    Args:
        metrics: List of metric dictionaries for different methods
        labels: List of method labels
        save_path: Path to save the plot (if None, plot is shown)
    """
    if not metrics:
        return
    
    metrics_to_plot = ['reprojection_error', 'point_density', 'coverage', 'occlusion_score', 'overall_score']
    num_metrics = len(metrics_to_plot)
    
    plt.figure(figsize=(15, 10))
    
    for i, metric_name in enumerate(metrics_to_plot):
        plt.subplot(2, 3, i + 1)
        
        values = [m.get(metric_name, 0) for m in metrics]
        plt.bar(labels, values)
        
        plt.title(metric_name.replace('_', ' ').title())
        plt.grid(True, alpha=0.3)
        
        # For error metrics, lower is better
        if 'error' in metric_name or 'occlusion' in metric_name:
            best_idx = np.argmin(values)
            plt.bar(labels[best_idx], values[best_idx], color='green')
        else:
            # For other metrics, higher is better
            best_idx = np.argmax(values)
            plt.bar(labels[best_idx], values[best_idx], color='green')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def log_training_metrics(episode: int, reward: float, metrics: Dict[str, float], selected_cameras: List[int], log_file: str):
    """
    Log training metrics to file.
    
    Args:
        episode: Current episode number
        reward: Episode reward
        metrics: Reconstruction quality metrics
        selected_cameras: List of selected camera indices
        log_file: Path to log file
    """
    metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    
    with open(log_file, 'a') as f:
        f.write(f"Episode {episode}, Reward: {reward:.4f}, Selected cameras: {selected_cameras}, Metrics: {metrics_str}\n")


def create_results_dir(base_dir: str = "results") -> str:
    """
    Create a timestamped directory for storing results.
    
    Args:
        base_dir: Base directory path
        
    Returns:
        Path to created directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Create subdirectories
    os.makedirs(os.path.join(results_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "logs"), exist_ok=True)
    
    return results_dir


def calculate_view_importance(agent, env) -> List[Tuple[int, float]]:
    """
    Calculate the importance of each camera view based on Q-values.
    
    Args:
        agent: Trained DQN agent
        env: Camera selection environment
        
    Returns:
        List of (camera_index, importance_score) tuples, sorted by importance
    """
    # Reset environment to get initial state
    state = env.reset()
    
    # Get Q-values for all actions in the initial state
    q_values = agent.qnetwork_local(torch.from_numpy(state).float().unsqueeze(0).to(agent.device)).cpu().detach().numpy()[0]
    
    # Create list of (camera_index, q_value) pairs
    view_importance = [(i, q_values[i]) for i in range(len(q_values))]
    
    # Sort by Q-value in descending order
    view_importance.sort(key=lambda x: x[1], reverse=True)
    
    return view_importance