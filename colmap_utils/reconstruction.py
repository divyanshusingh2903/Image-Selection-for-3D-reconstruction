import os
import subprocess
import numpy as np
import sqlite3
import pandas as pd
from typing import List, Dict, Tuple, Any


class COLMAPInterface:
    """Interface for running COLMAP and accessing reconstruction results."""
    
    def __init__(self, image_dir: str, output_dir: str, camera_positions_file: str = None):
        """
        Initialize COLMAP interface.
        
        Args:
            image_dir: Directory containing input images
            output_dir: Directory where COLMAP will store output files
            camera_positions_file: Optional file containing camera positions
        """
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.database_path = os.path.join(output_dir, "database.db")
        self.sparse_dir = os.path.join(output_dir, "sparse")
        self.dense_dir = os.path.join(output_dir, "dense")
        self.camera_positions = None
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.sparse_dir, exist_ok=True)
        os.makedirs(self.dense_dir, exist_ok=True)
        
        # Load camera positions if provided
        if camera_positions_file and os.path.exists(camera_positions_file):
            self.camera_positions = self._load_camera_positions(camera_positions_file)
    
    def _load_camera_positions(self, file_path: str) -> Dict[str, np.ndarray]:
        """
        Load camera positions from file.
        
        Args:
            file_path: Path to camera positions file
            
        Returns:
            Dictionary mapping image names to camera positions
        """
        camera_positions = {}
        
        # Assuming the file format is: image_name x y z [qw qx qy qz]
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:  # At minimum need image_name and x,y,z
                    image_name = parts[0]
                    position = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                    camera_positions[image_name] = position
        
        return camera_positions
    
    def run_feature_extraction(self) -> bool:
        """
        Run COLMAP feature extraction on input images.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            cmd = [
                "colmap", "feature_extractor",
                "--database_path", self.database_path,
                "--image_path", self.image_dir
            ]
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError:
            print("Error running COLMAP feature extraction")
            return False
    
    def run_feature_matching(self) -> bool:
        """
        Run COLMAP feature matching on extracted features.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            cmd = [
                "colmap", "exhaustive_matcher",
                "--database_path", self.database_path
            ]
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError:
            print("Error running COLMAP feature matching")
            return False
    
    def run_sparse_reconstruction(self) -> bool:
        """
        Run COLMAP sparse reconstruction (Structure from Motion).
        
        Returns:
            True if successful, False otherwise
        """
        try:
            cmd = [
                "colmap", "mapper",
                "--database_path", self.database_path,
                "--image_path", self.image_dir,
                "--output_path", self.sparse_dir
            ]
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError:
            print("Error running COLMAP sparse reconstruction")
            return False
    
    def run_dense_reconstruction(self) -> bool:
        """
        Run COLMAP dense reconstruction (Multi-View Stereo).
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Image undistortion
            cmd = [
                "colmap", "image_undistorter",
                "--image_path", self.image_dir,
                "--input_path", os.path.join(self.sparse_dir, "0"),
                "--output_path", self.dense_dir,
                "--output_type", "COLMAP"
            ]
            subprocess.run(cmd, check=True)
            
            # Dense stereo
            cmd = [
                "colmap", "patch_match_stereo",
                "--workspace_path", self.dense_dir
            ]
            subprocess.run(cmd, check=True)
            
            # Dense fusion
            cmd = [
                "colmap", "stereo_fusion",
                "--workspace_path", self.dense_dir,
                "--output_path", os.path.join(self.dense_dir, "fused.ply")
            ]
            subprocess.run(cmd, check=True)
            
            return True
        except subprocess.CalledProcessError:
            print("Error running COLMAP dense reconstruction")
            return False
    
    def full_reconstruction(self) -> bool:
        """
        Run full reconstruction pipeline: feature extraction, matching, sparse and dense reconstruction.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.run_feature_extraction():
            return False
        
        if not self.run_feature_matching():
            return False
        
        if not self.run_sparse_reconstruction():
            return False
        
        if not self.run_dense_reconstruction():
            return False
        
        return True
    
    def subset_reconstruction(self, image_indices: List[int]) -> Dict[str, Any]:
        """
        Run reconstruction using only a subset of images.
        
        Args:
            image_indices: List of indices of images to use
            
        Returns:
            Dictionary containing reconstruction quality metrics
        """
        # Create temporary directory for this subset reconstruction
        subset_dir = os.path.join(self.output_dir, f"subset_{'-'.join(map(str, image_indices))}")
        os.makedirs(subset_dir, exist_ok=True)
        
        # Get image paths for selected indices
        all_images = sorted(os.listdir(self.image_dir))
        selected_images = [all_images[i] for i in image_indices if i < len(all_images)]
        
        # Create symlinks to selected images in temporary directory
        subset_image_dir = os.path.join(subset_dir, "images")
        os.makedirs(subset_image_dir, exist_ok=True)
        
        for img in selected_images:
            src = os.path.join(self.image_dir, img)
            dst = os.path.join(subset_image_dir, img)
            if not os.path.exists(dst):
                os.symlink(src, dst)
        
        # Run reconstruction on subset
        subset_interface = COLMAPInterface(subset_image_dir, subset_dir)
        success = subset_interface.full_reconstruction()
        
        if not success:
            return {"success": False}
        
        # Calculate reconstruction quality metrics
        from .metrics import calculate_reconstruction_quality
        metrics = calculate_reconstruction_quality(subset_dir)
        metrics["success"] = True
        metrics["num_images"] = len(selected_images)
        
        return metrics
    
    def get_camera_info(self) -> List[Dict[str, Any]]:
        """
        Get information about cameras used in the reconstruction.
        
        Returns:
            List of dictionaries containing camera information
        """
        # Connect to the COLMAP database
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Get cameras
            cursor.execute("SELECT camera_id, model, width, height, params FROM cameras")
            cameras = {}
            for row in cursor.fetchall():
                camera_id, model, width, height, params = row
                params = np.frombuffer(params, dtype=np.float64)
                cameras[camera_id] = {
                    "model": model,
                    "width": width,
                    "height": height,
                    "params": params
                }
            
            # Get images and their camera relationships
            cursor.execute("SELECT image_id, name, camera_id FROM images")
            images = {}
            for row in cursor.fetchall():
                image_id, name, camera_id = row
                images[image_id] = {
                    "name": name,
                    "camera_id": camera_id
                }
            
            # Combine information
            camera_info = []
            for image_id, image_data in images.items():
                camera_id = image_data["camera_id"]
                camera_data = cameras.get(camera_id, {})
                
                info = {
                    "image_id": image_id,
                    "image_name": image_data["name"],
                    "camera_id": camera_id,
                    "model": camera_data.get("model"),
                    "width": camera_data.get("width"),
                    "height": camera_data.get("height")
                }
                
                # Add position if available
                if self.camera_positions and image_data["name"] in self.camera_positions:
                    info["position"] = self.camera_positions[image_data["name"]]
                
                camera_info.append(info)
            
            conn.close()
            return camera_info
            
        except Exception as e:
            print(f"Error retrieving camera information: {e}")
            return []
    
    def get_reconstruction_points(self) -> np.ndarray:
        """
        Get 3D points from the reconstruction.
        
        Returns:
            Array of 3D points
        """
        try:
            # Parse the PLY file
            ply_path = os.path.join(self.dense_dir, "fused.ply")
            if not os.path.exists(ply_path):
                return np.array([])
            
            # Simple PLY parser for ASCII format (can be extended for binary formats)
            points = []
            with open(ply_path, 'r') as f:
                header = True
                for line in f:
                    if header:
                        if "end_header" in line:
                            header = False
                        continue
                    
                    # Parse point coordinates (x, y, z)
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        points.append([float(parts[0]), float(parts[1]), float(parts[2])])
            
            return np.array(points)
            
        except Exception as e:
            print(f"Error retrieving reconstruction points: {e}")
            return np.array([])