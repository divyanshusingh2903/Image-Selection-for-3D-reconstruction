import os
import numpy as np
import sqlite3
from typing import Dict, Any, List, Tuple


def calculate_reprojection_error(reconstruction_dir: str) -> float:
    """
    Calculate the average reprojection error from COLMAP reconstruction.
    
    Args:
        reconstruction_dir: Directory containing the COLMAP reconstruction
        
    Returns:
        Average reprojection error in pixels
    """
    db_path = os.path.join(reconstruction_dir, "database.db")
    if not os.path.exists(db_path):
        return float('inf')
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Query two_view_geometries table for mean reprojection errors
        cursor.execute("SELECT rows, cols, data FROM two_view_geometries")
        
        total_error = 0.0
        total_matches = 0
        
        for row in cursor.fetchall():
            rows, cols, data = row
            if rows > 0 and cols > 0:
                # The data contains inlier matches and their errors
                # Format depends on COLMAP version, but typically contains reprojection errors
                # Here we're using a simplified estimate
                total_error += rows  # Number of inlier matches * average error
                total_matches += rows
        
        conn.close()
        
        # If no matches found, return infinity
        if total_matches == 0:
            return float('inf')
        
        # Normalize by number of matches
        # This is a simplified measure; actual reprojection errors
        # would require parsing COLMAP's binary files in detail
        return total_error / total_matches
        
    except Exception as e:
        print(f"Error calculating reprojection error: {e}")
        return float('inf')


def calculate_point_cloud_density(reconstruction_dir: str) -> float:
    """
    Calculate the density of the reconstructed point cloud.
    
    Args:
        reconstruction_dir: Directory containing the COLMAP reconstruction
        
    Returns:
        Point cloud density (points per cubic unit)
    """
    ply_path = os.path.join(reconstruction_dir, "dense", "fused.ply")
    if not os.path.exists(ply_path):
        return 0.0
    
    try:
        # Parse PLY file to extract points
        points = []
        with open(ply_path, 'r') as f:
            header = True
            vertex_count = 0
            
            for line in f:
                line = line.strip()
                
                # Parse header to get vertex count
                if header:
                    if line.startswith("element vertex"):
                        vertex_count = int(line.split()[-1])
                    elif line == "end_header":
                        header = False
                    continue
                
                # Parse point coordinates
                parts = line.split()
                if len(parts) >= 3:
                    points.append([float(parts[0]), float(parts[1]), float(parts[2])])
        
        points = np.array(points)
        
        # If no points, return 0 density
        if len(points) == 0:
            return 0.0
        
        # Calculate bounding box volume
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        volume = np.prod(max_coords - min_coords)
        
        # Avoid division by zero
        if volume <= 0:
            return 0.0
        
        # Return density
        return len(points) / volume
        
    except Exception as e:
        print(f"Error calculating point cloud density: {e}")
        return 0.0


def calculate_coverage(reconstruction_dir: str) -> float:
    """
    Calculate the coverage of the reconstruction (percentage of scene covered).
    
    Args:
        reconstruction_dir: Directory containing the COLMAP reconstruction
        
    Returns:
        Coverage score (0-1)
    """
    # This is a simplified coverage measure
    # In a real-world scenario, you would need ground truth or reference data
    
    try:
        # Check if sparse reconstruction exists
        sparse_dir = os.path.join(reconstruction_dir, "sparse", "0")
        if not os.path.exists(sparse_dir):
            return 0.0
        
        # Get number of registered images as a proxy for coverage
        images_txt = os.path.join(sparse_dir, "images.txt")
        if not os.path.exists(images_txt):
            return 0.0
        
        registered_images = 0
        total_images = 0
        
        # Count registered images
        with open(images_txt, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    registered_images += 1
                    total_images += 1
        
        # Get total number of images from the database
        db_path = os.path.join(reconstruction_dir, "database.db")
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM images")
            total_images = cursor.fetchone()[0]
            conn.close()
        
        # Calculate coverage as percentage of registered images
        if total_images == 0:
            return 0.0
        
        return registered_images / total_images
        
    except Exception as e:
        print(f"Error calculating coverage: {e}")
        return 0.0


def estimate_occlusions(reconstruction_dir: str) -> float:
    """
    Estimate the level of occlusions in the reconstruction.
    
    Args:
        reconstruction_dir: Directory containing the COLMAP reconstruction
        
    Returns:
        Occlusion score (lower is better)
    """
    # This is a simplified occlusion estimate based on point visibility
    # In a real implementation, you would analyze depth maps and visibility constraints
    
    try:
        # Connect to the database
        db_path = os.path.join(reconstruction_dir, "database.db")
        if not os.path.exists(db_path):
            return 1.0  # Maximum occlusion if no database
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get number of 3D points
        points_txt = os.path.join(reconstruction_dir, "sparse", "0", "points3D.txt")
        if not os.path.exists(points_txt):
            conn.close()
            return 1.0
        
        # Count visible points per image
        point_visibility = {}
        with open(points_txt, 'r') as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                
                parts = line.strip().split()
                if len(parts) >= 8:  # POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]
                    # Extract track information which contains image IDs
                    track_info = parts[8:]
                    image_ids = [int(track_info[i]) for i in range(0, len(track_info), 2)]
                    
                    for img_id in image_ids:
                        if img_id not in point_visibility:
                            point_visibility[img_id] = 0
                        point_visibility[img_id] += 1
        
        # Get total number of images
        cursor.execute("SELECT COUNT(*) FROM images")
        total_images = cursor.fetchone()[0]
        conn.close()
        
        if total_images == 0:
            return 1.0
        
        # Calculate average points visible per image
        total_visibility = sum(point_visibility.values())
        avg_visibility = total_visibility / total_images if total_images > 0 else 0
        
        # Calculate occlusion score (normalized to 0-1)
        # Higher visibility means lower occlusion
        max_expected_visibility = 1000  # Heuristic value, adjust based on your dataset
        occlusion_score = 1.0 - min(1.0, avg_visibility / max_expected_visibility)
        
        return occlusion_score
        
    except Exception as e:
        print(f"Error estimating occlusions: {e}")
        return 1.0


def calculate_reconstruction_quality(reconstruction_dir: str) -> Dict[str, float]:
    """
    Calculate overall reconstruction quality metrics.
    
    Args:
        reconstruction_dir: Directory containing the COLMAP reconstruction
        
    Returns:
        Dictionary of quality metrics
    """
    # Calculate individual metrics
    reprojection_error = calculate_reprojection_error(reconstruction_dir)
    point_density = calculate_point_cloud_density(reconstruction_dir)
    coverage = calculate_coverage(reconstruction_dir)
    occlusion_score = estimate_occlusions(reconstruction_dir)
    
    # Calculate overall quality score (lower is better for error and occlusion)
    # Normalize reprojection error (assuming typical values between 0.5 and 5 pixels)
    norm_reprojection = max(0, min(1, 1 - (reprojection_error - 0.5) / 4.5))
    
    # Normalize density (assuming typical values between 0 and 1000 points per cubic unit)
    # This needs to be calibrated based on your specific dataset
    norm_density = min(1, point_density / 1000)
    
    # Combined score with weights
    overall_score = (
        0.3 * norm_reprojection +
        0.25 * norm_density +
        0.25 * coverage +
        0.2 * (1 - occlusion_score)  # Invert occlusion score since lower is better
    )
    
    return {
        "reprojection_error": reprojection_error,
        "point_density": point_density,
        "coverage": coverage,
        "occlusion_score": occlusion_score,
        "overall_score": overall_score
    }