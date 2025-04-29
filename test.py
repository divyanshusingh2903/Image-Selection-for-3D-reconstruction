import os
import argparse
import numpy as np
import torch
import json
import random
from tqdm import tqdm
import matplotlib.pyplot as plt # Added for plotting
import pandas as pd # Added for data handling

from reinforcement_learning.environment import CameraSelectionEnv
from reinforcement_learning.agent import DQNAgent
from reinforcement_learning.utils import plot_metric_comparison, create_results_dir


def run_episode(env, agent=None, method='dqn', max_views=3, action_size=None):
    """Runs a single episode for evaluation."""
    state = env.reset()
    selected_cameras = []
    total_reward = 0
    final_metrics = {}
    done = False
    step_count = 0

    while not done and step_count < max_views:
        if method == 'dqn':
            if agent is None:
                raise ValueError("Agent must be provided for DQN method.")
            # Use epsilon=0 for deterministic behavior during testing
            action = agent.act(state, eps=0.0) 
        elif method == 'random':
            if action_size is None:
                 raise ValueError("Action size must be provided for random method.")
            # Ensure the chosen action hasn't been selected before in this episode
            available_actions = [i for i in range(action_size) if i not in selected_cameras]
            if not available_actions: # Should not happen if max_views <= num_cameras
                 break
            action = random.choice(available_actions)
        elif method == 'all':
             # Select cameras sequentially until max_views is reached or all cameras are selected
             if step_count < action_size:
                 action = step_count # Select camera index 'step_count'
                 if action in selected_cameras: # Should not happen with this logic
                     break
             else:
                 break # All cameras selected or max_views reached
        else:
            raise ValueError(f"Unknown method: {method}")

        # Ensure the selected action is valid within the environment's action space
        if action < 0 or action >= env.action_space.n:
           print(f"Warning: Invalid action {action} selected by method {method}. Skipping step.")
           # Decide how to handle this: skip, break, or choose randomly? Let's skip.
           step_count += 1 # Still counts as a step even if invalid
           continue


        next_state, reward, done, info = env.step(action)
        
        # Check if the action was valid within the environment logic (e.g., not already selected)
        if info.get("error") == "Camera already selected" and method != 'dqn':
             # If random or 'all' somehow selects an already selected camera, try again or break
             # This indicates a potential logic issue in the selection or environment state handling
             print(f"Warning: Method {method} selected an already chosen camera {action}. Breaking episode.")
             break # Avoid infinite loops
        elif info.get("error") == "Camera already selected" and method == 'dqn':
             # Agent might learn to select same camera; treat as no-op or penalize implicitly by reward
             # The environment already gives a negative reward for this.
             pass # Let the environment handle the reward/state


        if info.get("error") != "Camera already selected":
            state = next_state
            total_reward += reward
            selected_cameras = info.get("selected_cameras", selected_cameras) # Update selected cameras from info
            final_metrics = info.get("metrics", {})
            step_count += 1
        else:
             # If camera was already selected, don't increment step_count representing successful selections
             # Let the loop continue to potentially select a different camera if needed (for DQN)
             # or handle based on method logic above.
             pass


        # Ensure done is updated based on step_count reaching max_views
        if step_count >= max_views:
            done = True


    # Use reference metrics if the episode finished early or failed
    if not final_metrics.get("success", False):
         final_metrics = info.get("reference_metrics", {"overall_score": 0.0, "reprojection_error": float('inf')})


    return selected_cameras, total_reward, final_metrics


def test_agent(args):
    """
    Test the trained DQN agent against baselines.
    
    Args:
        args: Command line arguments
    """
    # --- Initialization ---
    # Create results directory for testing outputs
    test_results_dir = create_results_dir(os.path.join(args.results_dir, "test_results"))
    print(f"Saving testing results to {test_results_dir}")


    # Initialize environment
    env = CameraSelectionEnv(
        image_dir=args.image_dir,
        output_dir=args.output_dir, # Use a specific subdir for testing reconstructions
        camera_positions_file=args.camera_positions,
        max_views=args.max_views,
        computational_cost_weight=args.cost_weight,
        reset_reconstruction=False # Typically false during testing to reuse reconstructions
    )
    
    # Determine device
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_gpu else "cpu")
    print(f"Using device: {device}")
    
    # Initialize agent
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Make sure hidden_size is provided if needed by DQNAgent constructor
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        hidden_size=args.hidden_size, # Use hidden_size from args
         # Provide other necessary parameters, potentially loading from training config
        learning_rate=args.learning_rate, 
        gamma=args.gamma, 
        tau=args.tau, 
        buffer_size=args.buffer_size, 
        batch_size=args.batch_size, 
        update_every=args.update_every, 
        device=device
    )
    
    # Load trained model weights
    if args.model_path and os.path.exists(args.model_path):
        agent.load(args.model_path)
        print(f"Loaded trained model from: {args.model_path}")
    else:
        print("Warning: No trained model path provided or path invalid. Testing with an untrained agent.")
        # Optionally: raise an error if a trained model is strictly required
        # raise FileNotFoundError("Trained model not found at specified path.")

    # --- Evaluation ---
    methods = ['dqn', 'random', 'all'] # Methods to compare [cite: 7]
    results = {method: {'rewards': [], 'metrics': [], 'selections': []} for method in methods}
    num_test_episodes = args.num_test_episodes

    for method in methods:
        print(f"\nEvaluating method: {method}...")
        for episode in tqdm(range(num_test_episodes), desc=f"Testing {method}"):
            if method == 'dqn':
                selected_cameras, total_reward, final_metrics = run_episode(
                    env, agent=agent, method='dqn', max_views=args.max_views
                )
            elif method == 'random':
                 selected_cameras, total_reward, final_metrics = run_episode(
                    env, agent=None, method='random', max_views=args.max_views, action_size=action_size
                )
            elif method == 'all':
                # The 'all' method attempts to select up to max_views sequentially
                # or all cameras if num_cameras < max_views.
                selected_cameras, total_reward, final_metrics = run_episode(
                    env, agent=None, method='all', max_views=args.max_views, action_size=action_size
                )

            results[method]['rewards'].append(total_reward)
            results[method]['metrics'].append(final_metrics)
            results[method]['selections'].append(selected_cameras)

    # --- Analysis & Saving ---
    print("\n--- Evaluation Summary ---")
    summary_data = []
    for method in methods:
        avg_reward = np.mean(results[method]['rewards'])
        # Calculate average for key metrics, handle potential missing keys or non-numeric data gracefully
        avg_quality_score = np.mean([m.get('overall_score', 0) for m in results[method]['metrics'] if m])
        avg_reproj_error = np.mean([m.get('reprojection_error', float('inf')) for m in results[method]['metrics'] if m])
        avg_views = np.mean([len(s) for s in results[method]['selections']])
        
        print(f"Method: {method}")
        print(f"  Avg Reward: {avg_reward:.4f}")
        print(f"  Avg Overall Score: {avg_quality_score:.4f}")
        print(f"  Avg Reprojection Error: {avg_reproj_error:.4f}")
        print(f"  Avg Views Selected: {avg_views:.2f}")

        summary_data.append({
            'Method': method,
            'Avg Reward': avg_reward,
            'Avg Overall Score': avg_quality_score,
            'Avg Reprojection Error': avg_reproj_error,
            'Avg Views Selected': avg_views
        })

    # Save detailed results
    results_path = os.path.join(test_results_dir, "test_results.json")
    # Convert numpy types for JSON serialization if necessary
    serializable_results = {}
    for method, data in results.items():
         serializable_results[method] = {
             'rewards': [float(r) for r in data['rewards']],
             # Ensure metrics are serializable (they likely are dicts of floats)
             'metrics': data['metrics'], 
             'selections': data['selections'] # List of lists of ints
         }

    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=4)
    print(f"Detailed results saved to {results_path}")

    # Save summary results
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(test_results_dir, "test_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary results saved to {summary_path}")


    # --- Plotting ---
    # Example: Plotting average reward comparison
    plt.figure(figsize=(10, 6))
    plt.bar(summary_df['Method'], summary_df['Avg Reward'])
    plt.title('Average Reward Comparison')
    plt.ylabel('Average Reward')
    plt.xlabel('Method')
    plot_path = os.path.join(test_results_dir, "average_reward_comparison.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Reward comparison plot saved to {plot_path}")


    # Example: Plotting average overall score comparison
    plt.figure(figsize=(10, 6))
    plt.bar(summary_df['Method'], summary_df['Avg Overall Score'])
    plt.title('Average Overall Score Comparison')
    plt.ylabel('Average Score')
    plt.xlabel('Method')
    plot_path = os.path.join(test_results_dir, "average_score_comparison.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Score comparison plot saved to {plot_path}")


    print("\nTesting finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test DQN agent for camera view selection")
    
    # Environment settings (similar to train.py)
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for reconstruction outputs (testing specific)")
    parser.add_argument("--camera_positions", type=str, default=None, help="File containing camera positions")
    parser.add_argument("--max_views", type=int, default=3, help="Maximum number of views to select per episode")
    parser.add_argument("--cost_weight", type=float, default=0.1, help="Weight for computational cost (should match training)")


     # Agent settings - Required for initializing the agent structure before loading weights
    parser.add_argument("--hidden_size", type=int, default=128, help="Size of hidden layers in Q-network (must match trained model)")
    # Add other DQNAgent parameters if they are needed for initialization before loading
    # These defaults should ideally match the training config, or be loaded from it
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate (not used for testing, but needed for init)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor (not used for testing, but needed for init)")
    parser.add_argument("--tau", type=float, default=1e-3, help="Soft update parameter (not used for testing, but needed for init)")
    parser.add_argument("--buffer_size", type=int, default=10000, help="Replay buffer size (not used for testing, but needed for init)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (not used for testing, but needed for init)")
    parser.add_argument("--update_every", type=int, default=4, help="Update frequency (not used for testing, but needed for init)")


    # Testing specific settings
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained DQN model checkpoint (.pth file)")
    parser.add_argument("--num_test_episodes", type=int, default=100, help="Number of episodes to run for testing")
    parser.add_argument("--results_dir", type=str, default="results", help="Base directory for saving testing results")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for testing if available")


    
    args = parser.parse_args()


    # config_path = os.path.join(os.path.dirname(args.model_path), "../", "config.json") # Example path structure
    # try:
    #     with open(config_path, 'r') as f:
    #         train_config = json.load(f)
    #     # Override args with training config where necessary (e.g., hidden_size)
    #     args.hidden_size = train_config.get('hidden_size', args.hidden_size)
    #     args.max_views = train_config.get('max_views', args.max_views)
    #     # ... potentially others like cost_weight ...
    # except FileNotFoundError:
    #     print(f"Warning: Training config not found at {config_path}. Using command line args.")


    test_agent(args)