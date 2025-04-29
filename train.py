import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

from reinforcement_learning.environment import CameraSelectionEnv
from reinforcement_learning.agent import DQNAgent
from reinforcement_learning.utils import (
    linear_decay_epsilon, 
    plot_rewards, 
    log_training_metrics, 
    create_results_dir,
    calculate_view_importance
)


def train(args):
    """
    Train the DQN agent for optimal camera view selection.
    
    Args:
        args: Command line arguments
    """
    # Create results directory
    results_dir = create_results_dir(args.results_dir)
    print(f"Saving results to {results_dir}")
    
    # Initialize environment
    env = CameraSelectionEnv(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        camera_positions_file=args.camera_positions,
        max_views=args.max_views,
        computational_cost_weight=args.cost_weight,
        reset_reconstruction=args.reset_reconstruction
    )
    
    # Determine device
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_gpu else "cpu")
    print(f"Using device: {device}")
    
    # Initialize agent
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        tau=args.tau,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        update_every=args.update_every,
        device=device
    )
    
    # Save configuration
    config = vars(args)
    config["state_size"] = state_size
    config["action_size"] = action_size
    
    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    
    # Training loop
    rewards = []
    log_file = os.path.join(results_dir, "logs", "training_log.txt")
    
    # Initialize progress bar
    progress_bar = tqdm(range(1, args.num_episodes + 1), desc="Training")
    
    for episode in progress_bar:
        # Reset environment
        state = env.reset()
        episode_reward = 0
        
        # Calculate epsilon
        eps = linear_decay_epsilon(
            episode, args.num_episodes, 
            eps_start=args.eps_start, 
            eps_end=args.eps_end
        )
        
        # Episode loop
        done = False
        while not done:
            # Select and take action
            action = agent.act(state, eps)
            next_state, reward, done, info = env.step(action)
            
            # Store experience and learn
            agent.step(state, action, reward, next_state, done)
            
            # Update state and reward
            state = next_state
            episode_reward += reward
        
        # Log metrics
        rewards.append(episode_reward)
        log_training_metrics(
            episode=episode,
            reward=episode_reward,
            metrics=info["metrics"],
            selected_cameras=info["selected_cameras"],
            log_file=log_file
        )
        
        # Update progress bar
        progress_bar.set_postfix({
            "reward": f"{episode_reward:.2f}", 
            "epsilon": f"{eps:.2f}",
            "selected": f"{info['selected_cameras']}"
        })
        
        # Save model periodically
        if episode % args.save_freq == 0 or episode == args.num_episodes:
            model_path = os.path.join(results_dir, "models", f"dqn_episode_{episode}.pth")
            agent.save(model_path)
            
            # Plot rewards
            plot_path = os.path.join(results_dir, "plots", f"rewards_episode_{episode}.png")
            plot_rewards(rewards, avg_window=min(100, len(rewards)), save_path=plot_path)
    
    # Calculate view importance after training
    view_importance = calculate_view_importance(agent, env)
    
    # Save view importance scores
    with open(os.path.join(results_dir, "view_importance.json"), "w") as f:
        json.dump({str(idx): float(score) for idx, score in view_importance}, f, indent=4)
    
    print("Training complete!")
    print(f"Results saved to {results_dir}")
    
    # Return final trained agent and results directory
    return agent, results_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN agent for camera view selection")
    
    # Environment settings
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for reconstruction outputs")
    parser.add_argument("--camera_positions", type=str, default=None, help="File containing camera positions")
    parser.add_argument("--max_views", type=int, default=3, help="Maximum number of views to select")
    parser.add_argument("--cost_weight", type=float, default=0.1, help="Weight for computational cost in reward")
    parser.add_argument("--reset_reconstruction", action="store_true", help="Reset reconstruction between episodes")
    
    # Agent settings
    parser.add_argument("--hidden_size", type=int, default=128, help="Size of hidden layers in Q-network")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for optimizer")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--tau", type=float, default=1e-3, help="Soft update parameter")
    parser.add_argument("--buffer_size", type=int, default=10000, help="Size of replay buffer")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--update_every", type=int, default=4, help="How often to update the target network")
    
    # Training settings
    parser.add_argument("--num_episodes", type=int, default=1000, help="Number of episodes to train")
    parser.add_argument("--eps_start", type=float, default=1.0, help="Starting epsilon value")
    parser.add_argument("--eps_end", type=float, default=0.01, help="Ending epsilon value")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for training if available")
    parser.add_argument("--save_freq", type=int, default=100, help="How often to save model checkpoints")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory for saving results")
    
    args = parser.parse_args()
    
    # Train agent
    train(args)