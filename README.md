# Camera View Selection using Deep Reinforcement Learning for 3D Reconstruction

## Overview

This project develops a Deep Reinforcement Learning (DRL) agent to optimize the selection of camera views for multi-view 3D reconstruction, specifically targeting applications like sports analysis. The goal is to dynamically choose a subset of available camera views that maximizes the quality of the 3D reconstruction while minimizing the associated computational cost, which can be significant when using all available views.

## Problem Description

In scenarios with multiple cameras capturing an event (e.g., a soccer match), using every camera feed for 3D reconstruction is often computationally expensive and potentially redundant. Selecting an optimal subset of views can significantly reduce processing time without sacrificing reconstruction quality. This project frames the view selection process as a sequential decision-making problem solvable with reinforcement learning.

## Features

* **Reinforcement Learning Environment**: A custom Gym-compatible environment (`CameraSelectionEnv`) that simulates the process of selecting camera views and evaluating reconstruction quality.
* **COLMAP Integration**: Interfaces with COLMAP (`colmap_utils`) to perform 3D reconstruction based on selected views and calculate quality metrics.
* **DQN Agent**: Implements a Deep Q-Network (DQN) agent (`DQNAgent`) to learn the optimal view selection policy.
* **Training Pipeline**: Script (`train.py`) to train the DQN agent.
* **Evaluation Pipeline**: Script (`test.py`) to evaluate the trained agent against baseline methods (random selection, using all views).
* **Metrics Calculation**: Computes reconstruction quality metrics (e.g., reprojection error, point cloud density) to guide the learning process and evaluate performance.

## Dependencies

* Python 3.x
* PyTorch
* NumPy
* OpenCV (cv2) - Likely needed by COLMAP or image processing
* Gymnasium (or OpenAI Gym)
* Matplotlib
* tqdm
* pandas
* **COLMAP**: Must be installed and accessible in your system PATH. (Refer to COLMAP installation guide: https://colmap.github.io/install.html)

Install Python dependencies using pip:
```bash
pip install -r requirements.txt
```
## Setup

- **Install COLMAP**: Follow the official installation instructions for your operating system. Ensure the colmap command is accessible from your terminal.
Install Python dependencies:
Bash

- pip install torch numpy gym matplotlib tqdm pandas opencv-python

```bash
python train.py \
    --image_dir data/images/ \
    --output_dir colmap_output_train/ \
    --results_dir results/ \
    --max_views 5 \
    --num_episodes 2000 \
    --cost_weight 0.05 \
    --hidden_size 256 \
```

## Testing
Run the test.py script to evaluate a trained model against baselines. Key arguments include:

 - **--image_dir**: Path to the directory containing input images.
 - **--output_dir**: Path to a directory where COLMAP will store reconstruction outputs during testing.
 - **--model_path**: Required. Path to the saved model checkpoint (.pth file) from training (e.g., results/experiment_XYZ/models/dqn_episode_2000.pth).
 - **--results_dir**: Path to the base directory where test results will be saved (within a test_results subdirectory).
 - **--num_test_episodes**: Number of episodes to run for evaluation.
 - **--max_views**: Maximum views to select (should match training).
 - **--hidden_size**: Network hidden size (must match the trained model).
Example:


## Reinforcement Learning Details
**State Space**: The state includes:
Features for each camera (e.g., position if available, potentially image-based features).
Binary indicator for whether each camera has been selected.
Current reconstruction quality metrics (e.g., reprojection error, point density, coverage).

**Action Space**: Discrete space representing the selection of one of the available camera views that has not yet been selected.

**Reward Function**: Designed to balance reconstruction quality and cost:
 - Positive reward based on the improvement in reconstruction quality (e.g., using an overall score derived from metrics) after adding a view.
 - Negative reward proportional to the computational cost (e.g., number of views selected).

Potential terminal bonus based on final quality relative to using all views.

**Agent**: Deep Q-Network (DQN) with experience replay and a target network for stable learning.

## Baselines
The performance of the DQN agent is compared against:

**Random Selection**: Randomly selecting available cameras up to max_views.

**All Views (Sequential)**: 
Sequentially selecting cameras (view 0, view 1, ...) up to max_views. If max_views is greater than or equal to the total number of cameras, this effectively uses all cameras within the episode limit. The reference reconstruction using all cameras simultaneously is often computed once for reward calculation and final comparison.
