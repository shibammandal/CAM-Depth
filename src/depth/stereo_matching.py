"""
Stereo depth estimation using OpenCV block matching.
Parameters are configurable and can be optimized by RL.
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional
import yaml


class StereoDepthEstimator:
    """
    Stereo depth estimation using Semi-Global Block Matching (SGBM).
    Parameters can be adjusted and are optimized by the RL agent.
    """
    
    # Parameter bounds for RL optimization
    PARAM_BOUNDS = {
        'block_size': (3, 21, 2),  # (min, max, step) - must be odd
        'min_disparity': (0, 32, 1),
        'num_disparities': (16, 128, 16),  # must be divisible by 16
        'uniqueness_ratio': (0, 20, 1),
        'speckle_window_size': (50, 200, 10),
        'speckle_range': (1, 5, 1),
        'disp12_max_diff': (1, 10, 1),
        'p1_multiplier': (1, 16, 1),  # P1 = multiplier * block_size^2
        'p2_multiplier': (1, 64, 1),  # P2 = multiplier * block_size^2
    }
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize stereo depth estimator with default parameters.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        
        depth_cfg = self.config.get('depth_estimation', {})
        
        # Initialize parameters (these are what RL will optimize)
        self.params = {
            'block_size': depth_cfg.get('block_size', 5),
            'min_disparity': depth_cfg.get('min_disparity', 0),
            'num_disparities': depth_cfg.get('num_disparities', 64),
            'uniqueness_ratio': depth_cfg.get('uniqueness_ratio', 10),
            'speckle_window_size': depth_cfg.get('speckle_window_size', 100),
            'speckle_range': depth_cfg.get('speckle_range', 2),
            'disp12_max_diff': 1,
            'p1_multiplier': 8,
            'p2_multiplier': 32,
        }
        
        # Camera intrinsics (set from stereo camera)
        self.focal_length = None
        self.baseline = None
        
        # Create stereo matcher
        self._update_matcher()
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {'depth_estimation': {}}
    
    def _update_matcher(self):
        """Update the stereo matcher with current parameters."""
        block_size = self.params['block_size']
        
        # Ensure block_size is odd
        if block_size % 2 == 0:
            block_size += 1
            self.params['block_size'] = block_size
        
        # P1 and P2 for SGBM
        p1 = self.params['p1_multiplier'] * block_size ** 2
        p2 = self.params['p2_multiplier'] * block_size ** 2
        
        self.stereo_matcher = cv2.StereoSGBM_create(
            minDisparity=self.params['min_disparity'],
            numDisparities=self.params['num_disparities'],
            blockSize=block_size,
            P1=p1,
            P2=p2,
            disp12MaxDiff=self.params['disp12_max_diff'],
            uniquenessRatio=self.params['uniqueness_ratio'],
            speckleWindowSize=self.params['speckle_window_size'],
            speckleRange=self.params['speckle_range'],
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
    
    def set_camera_params(self, focal_length: float, baseline: float):
        """
        Set camera parameters for depth calculation.
        
        Args:
            focal_length: Camera focal length in pixels
            baseline: Distance between stereo cameras in meters
        """
        self.focal_length = focal_length
        self.baseline = baseline
    
    def set_params(self, params: Dict):
        """
        Update stereo matching parameters.
        
        Args:
            params: Dictionary of parameter values
        """
        for key, value in params.items():
            if key in self.params:
                # Clip to valid bounds
                if key in self.PARAM_BOUNDS:
                    min_val, max_val, step = self.PARAM_BOUNDS[key]
                    value = np.clip(value, min_val, max_val)
                    # Round to nearest step
                    value = round((value - min_val) / step) * step + min_val
                    value = int(value)
                self.params[key] = value
        
        self._update_matcher()
    
    def get_params(self) -> Dict:
        """Get current parameters."""
        return self.params.copy()
    
    def get_param_vector(self) -> np.ndarray:
        """
        Get parameters as normalized vector for RL.
        
        Returns:
            Normalized parameter vector in [0, 1] range
        """
        vector = []
        for key in sorted(self.PARAM_BOUNDS.keys()):
            min_val, max_val, _ = self.PARAM_BOUNDS[key]
            normalized = (self.params[key] - min_val) / (max_val - min_val)
            vector.append(normalized)
        return np.array(vector, dtype=np.float32)
    
    def set_param_vector(self, vector: np.ndarray):
        """
        Set parameters from normalized vector.
        
        Args:
            vector: Normalized parameter vector (values in [0, 1])
        """
        sorted_keys = sorted(self.PARAM_BOUNDS.keys())
        for i, key in enumerate(sorted_keys):
            min_val, max_val, step = self.PARAM_BOUNDS[key]
            # Denormalize
            value = vector[i] * (max_val - min_val) + min_val
            # Round to step
            value = round((value - min_val) / step) * step + min_val
            self.params[key] = int(value)
        
        self._update_matcher()
    
    @staticmethod
    def param_dim() -> int:
        """Get dimension of parameter vector."""
        return len(StereoDepthEstimator.PARAM_BOUNDS)
    
    def compute_disparity(self, left_image: np.ndarray, 
                          right_image: np.ndarray) -> np.ndarray:
        """
        Compute disparity map from stereo images.
        
        Args:
            left_image: Left camera image (RGB or grayscale)
            right_image: Right camera image (RGB or grayscale)
            
        Returns:
            Disparity map (float32)
        """
        # Convert to grayscale if needed
        if len(left_image.shape) == 3:
            left_gray = cv2.cvtColor(left_image, cv2.COLOR_RGB2GRAY)
        else:
            left_gray = left_image
            
        if len(right_image.shape) == 3:
            right_gray = cv2.cvtColor(right_image, cv2.COLOR_RGB2GRAY)
        else:
            right_gray = right_image
        
        # Compute disparity
        disparity = self.stereo_matcher.compute(left_gray, right_gray)
        
        # Convert to float32 and scale
        disparity = disparity.astype(np.float32) / 16.0
        
        return disparity
    
    def compute_depth(self, left_image: np.ndarray,
                      right_image: np.ndarray) -> np.ndarray:
        """
        Compute depth map from stereo images.
        
        Args:
            left_image: Left camera image
            right_image: Right camera image
            
        Returns:
            Depth map in meters
        """
        if self.focal_length is None or self.baseline is None:
            raise ValueError("Camera parameters not set. Call set_camera_params first.")
        
        disparity = self.compute_disparity(left_image, right_image)
        
        # Avoid division by zero
        disparity[disparity <= 0] = 0.1
        
        # Depth = (focal_length * baseline) / disparity
        depth = (self.focal_length * self.baseline) / disparity
        
        # Clip to reasonable range
        depth = np.clip(depth, 0.1, 50.0)
        
        return depth
    
    def compute_depth_with_quality(self, left_image: np.ndarray,
                                   right_image: np.ndarray,
                                   ground_truth: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """
        Compute depth with quality metrics.
        
        Args:
            left_image: Left camera image
            right_image: Right camera image
            ground_truth: Optional ground truth depth for comparison
            
        Returns:
            (depth_map, metrics_dict)
        """
        depth = self.compute_depth(left_image, right_image)
        
        metrics = {
            'valid_pixels': np.sum(depth < 50.0) / depth.size,
            'mean_depth': np.mean(depth[depth < 50.0]) if np.any(depth < 50.0) else 0,
        }
        
        if ground_truth is not None:
            # Reshape ground_truth if it comes as flat array from PyBullet
            if ground_truth.ndim == 1:
                ground_truth = ground_truth.reshape(depth.shape)
            elif ground_truth.shape != depth.shape:
                # Try to reshape to match depth dimensions
                try:
                    ground_truth = ground_truth.reshape(depth.shape)
                except ValueError:
                    # If reshape fails, skip ground truth comparison
                    return depth, metrics
            
            # Mask for valid comparisons
            valid_mask = (depth < 50.0) & (ground_truth < 50.0) & (ground_truth > 0.1)
            
            if np.any(valid_mask):
                # Mean Absolute Error
                mae = np.mean(np.abs(depth[valid_mask] - ground_truth[valid_mask]))
                
                # Root Mean Square Error
                mse = np.mean((depth[valid_mask] - ground_truth[valid_mask]) ** 2)
                rmse = np.sqrt(mse)
                
                # Relative error
                rel_error = np.mean(np.abs(depth[valid_mask] - ground_truth[valid_mask]) / ground_truth[valid_mask])
                
                # Accuracy thresholds (within delta of ground truth)
                for delta in [1.05, 1.10, 1.25]:
                    ratio = np.maximum(depth[valid_mask] / ground_truth[valid_mask],
                                      ground_truth[valid_mask] / depth[valid_mask])
                    acc = np.mean(ratio < delta)
                    metrics[f'acc_{delta}'] = acc
                
                metrics['mae'] = mae
                metrics['rmse'] = rmse
                metrics['rel_error'] = rel_error
            else:
                metrics['mae'] = float('inf')
                metrics['rmse'] = float('inf')
                metrics['rel_error'] = float('inf')
        
        return depth, metrics
    
    def postprocess_depth(self, depth: np.ndarray) -> np.ndarray:
        """
        Apply post-processing to clean up depth map.
        
        Args:
            depth: Raw depth map
            
        Returns:
            Cleaned depth map
        """
        # Bilateral filter to smooth while preserving edges
        depth_uint16 = (depth * 100).astype(np.uint16)
        depth_filtered = cv2.bilateralFilter(
            depth_uint16.astype(np.float32), 
            d=5, 
            sigmaColor=75, 
            sigmaSpace=75
        )
        depth_filtered = depth_filtered / 100.0
        
        # Fill small holes with morphological operations
        kernel = np.ones((3, 3), np.uint8)
        mask = (depth_filtered > 49.0).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Inpaint holes
        if np.any(mask > 0):
            depth_inpainted = cv2.inpaint(
                depth_filtered.astype(np.float32),
                mask,
                inpaintRadius=3,
                flags=cv2.INPAINT_TELEA
            )
            return depth_inpainted
        
        return depth_filtered
