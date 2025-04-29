import os
import numpy as np
import gym
from gym import spaces
import torch
from typing import Dict, List, Tuple, Any, Optional

from colmap_utils.reconstruction import COLMAPInterface
from colmap_utils.metrics import calculate_reconstruction_quality


class CameraSelectionEnv(gym.Env):
    """
    Environment for training a DRL agent to select optimal camera views for 3D reconstruction.
    """
    
    def __init__(
        self,
        image_dir: str,
        output_dir: str,
        camera_positions_file: Optional[str] = None,
        max_views: int = 3,
        computational_cost_weight: float = 0.1,
        reset_reconstruction: bool = False
    ):
        """
        Initialize the environment.
        
        Args:
            image_dir: Directory containing the input images
            output_dir: Directory for storing reconstruction outputs
            camera_positions_file: File containing camera positions (optional)
            max_views: Maximum number of views to select
            computational_cost_weight: Weight for computational cost in reward
            reset_reconstruction: Whether to reset the reconstruction between episodes
        """
        super(CameraSelectionEnv, self).__init__()
        
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.camera_positions_file = camera_positions_file
        self.max_views = max_views
        self.computational_cost_weight = computational_cost_weight
        self.reset_reconstruction = reset_reconstruction
        
        # Create COLMAP interface
        self.colmap = COLMAPInterface(image_dir, output_dir, camera_positions_file)
        
        # Get available images and their count
        self.available_images = sorted(os.listdir(image_dir))
        self.num_cameras = len(self.available_images)
        
        # Action space: select one of the cameras
        self.action_space = spaces.Discrete(self.num_cameras)
        
        # State space: features for each camera view + reconstruction metrics + selected views
        # For each camera:
        # - 3D position (if available)
        # - Basic image features (simplified for now)
        # - Already selected (binary)
        
        # For reconstruction:
        # - Current reprojection error
        # - Current point cloud density
        # - Current coverage
        # - Current occlusion score
        
        # Determine state dimension
        camera_features_dim = 4  # Position (3D) + selected status
        if self.camera_positions_file:
            camera_features_dim = 7  # Position (3D) + extracted features (3D) + selected status
        
        reconstruction_metrics_dim = 4  # Reprojection error, density, coverage, occlusion
        
        state_dim = (self.num_cameras * camera_features_dim) + reconstruction_metrics_dim
        
        # Define state bounds
        low = np.ones(state_dim) * -np.inf
        high = np.ones(state_dim) * np.inf
        
        # Camera selection status is binary
        for i in range(self.num_cameras):
            low[i * camera_features_dim + (camera_features_dim - 1)] = 0
            high[i * camera_features_dim + (camera_features_dim - 1)] = 1
        
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        # Environment state
        self.selected_cameras = []
        self.current_step = 0
        self.current_metrics = None
        self.reference_metrics = None
        
        # Pre-compute image features (in a real implementation, this would include
        # meaningful features extracted from the images)
        self.image_features = self._extract_image_features()
    
    def _extract_image_features(self) -> np.ndarray:
        """
        Extract features from all available images.
        In a real implementation, this would extract meaningful features
        from the images using computer vision techniques.
        
        Returns:
            Array of features for each image
        """
        # Simplified features (random for now)
        # In a real implementation, you would extract real features from images
        features = np.random.rand(self.num_cameras, 3)
        
        # If camera positions are available, use them as features
        if self.camera_positions_file and self.colmap.camera_positions:
            camera_info = self.colmap.get_camera_info()
            for i, info in enumerate(camera_info):
                if "position" in info:
                    image_name = info["image_name"]
                    image_index = self.available_images.index(image_name) if image_name in self.available_images else -1
                    if image_index >= 0:
                        features[image_index] = info["position"]
        
        return features
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment to starting state.
        
        Returns:
            Initial state observation
        """
        # Reset selected cameras and current step
        self.selected_cameras = []
        self.current_step = 0
        self.current_metrics = None
        
        # Reset the reconstruction if needed
        if self.reset_reconstruction:
            # In a real implementation, you might want to reset the COLMAP database
            # or create a new output directory
            pass
        
        # Get reference metrics (using all views) if not already computed
        if self.reference_metrics is None:
            # Use all camera indices
            all_indices = list(range(self.num_cameras))
            self.reference_metrics = self.colmap.subset_reconstruction(all_indices)
        
        # Get initial observation (no cameras selected yet)
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment by selecting a camera view.
        
        Args:
            action: Camera index to select
            
        Returns:
            observation: New state observation
            reward: Reward for the action
            done: Whether the episode is done
            info: Additional information
        """
        # Validate action
        if action < 0 or action >= self.num_cameras:
            raise ValueError(f"Invalid action: {action}. Must be between 0 and {self.num_cameras-1}")
        
        # Check if camera already selected
        if action in self.selected_cameras:
            # Penalize selecting the same camera twice
            reward = -1.0
            done = False
            info = {"error": "Camera already selected"}
            return self._get_observation(), reward, done, info
        
        # Add selected camera to the list
        self.selected_cameras.append(action)
        self.current_step += 1
        
        # Run reconstruction with current subset of cameras
        previous_metrics = self.current_metrics
        self.current_metrics = self.colmap.subset_reconstruction(self.selected_cameras)
        
        # Calculate reward
        reward = self._calculate_reward(previous_metrics, self.current_metrics)
        
        # Check if episode is done
        done = self.current_step >= self.max_views
        
        # Additional info
        info = {
            "selected_cameras": self.selected_cameras,
            "metrics": self.current_metrics,
            "reference_metrics": self.reference_metrics
        }
        
        return self._get_observation(), reward, done, info
    
    def _calculate_reward(self, previous_metrics: Dict[str, float], current_metrics: Dict[str, float]) -> float:
        """
        Calculate reward based on improvement in reconstruction quality.
        
        Args:
            previous_metrics: Metrics before adding the new view
            current_metrics: Metrics after adding the new view
            
        Returns:
            Reward value
        """
        # Check if reconstruction was successful
        if not current_metrics.get("success", False):
            return -2.0  # Penalize failed reconstructions
        
        # If first camera, just use raw score
        if previous_metrics is None:
            quality_score = current_metrics.get("overall_score", 0.0)
            # Scale the score to provide appropriate reward signal
            return quality_score * 2.0
        
        # Calculate improvement in quality metrics
        prev_quality = previous_metrics.get("overall_score", 0.0)
        curr_quality = current_metrics.get("overall_score", 0.0)
        quality_improvement = curr_quality - prev_quality
        
        # Calculate computational cost (based on number of selected cameras)
        computational_cost = len(self.selected_cameras) / self.num_cameras
        
        # Calculate reward as quality improvement minus computational cost
        reward = (quality_improvement * 10.0) - (self.computational_cost_weight * computational_cost)
        
        # Reference comparison for terminal state
        if self.current_step == self.max_views:
            # Compare to reference metrics (using all views)
            ref_quality = self.reference_metrics.get("overall_score", 0.0)
            if ref_quality > 0:
                relative_quality = curr_quality / ref_quality
                # Bonus reward for achieving good quality with fewer views
                bonus = relative_quality * (1.0 - computational_cost) * 5.0
                reward += bonus
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """
        Get the current state observation.
        
        Returns:
            Current state observation as a numpy array
        """
        # Initialize state array
        camera_features_dim = 7 if self.camera_positions_file else 4
        state = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        # Set camera features
        for i in range(self.num_cameras):
            base_idx = i * camera_features_dim
            
            # Set camera position/features
            state[base_idx:base_idx+3] = self.image_features[i]
            
            # Set additional features if available
            if camera_features_dim > 4:
                # These would be image-specific features in a real implementation
                state[base_idx+3:base_idx+6] = np.random.rand(3)  # Placeholder
            
            # Set selection status
            state[base_idx+camera_features_dim-1] = 1.0 if i in self.selected_cameras else 0.0
        
        # Set reconstruction metrics
        reconstruction_offset = self.num_cameras * camera_features_dim
        
        if self.current_metrics is not None:
            state[reconstruction_offset] = self.current_metrics.get("reprojection_error", 0.0)
            state[reconstruction_offset+1] = self.current_metrics.get("point_density", 0.0)
            state[reconstruction_offset+2] = self.current_metrics.get("coverage", 0.0)
            state[reconstruction_offset+3] = self.current_metrics.get("occlusion_score", 0.0)
        
        return state
    
    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode: Rendering mode
        """
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Selected cameras: {self.selected_cameras}")
            if self.current_metrics:
                print(f"Current metrics: {self.current_metrics}")
        
        # Could add visualization of the reconstruction here
        
    def close(self):
        """
        Clean up resources.
        """
        pass