"""
Verification system for video washing effects.
Measures how effectively washing techniques change the video's digital signature
while maintaining visual similarity.
"""

import os
import cv2
import numpy as np
import subprocess
import logging
from typing import Dict, Tuple, List, Optional
import hashlib
import imagehash
from PIL import Image
import tempfile

logger = logging.getLogger(__name__)

class WashVerifier:
    """Verifies the effectiveness of video washing techniques."""
    
    def __init__(self, temp_dir: Optional[str] = None):
        """
        Initialize the wash verifier.
        
        Args:
            temp_dir: Optional directory for temporary files. If None, system default is used.
        """
        self.temp_dir = temp_dir or tempfile.gettempdir()
        
    def extract_frames(self, video_path: str, num_frames: int = 5) -> List[np.ndarray]:
        """
        Extract a sample of frames from the video for comparison.
        
        Args:
            video_path: Path to the video file
            num_frames: Number of frames to sample (evenly distributed)
            
        Returns:
            List of frame images as numpy arrays
        """
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames <= 0:
                logger.warning(f"Could not determine frame count for {video_path}")
                return []
                
            # Calculate frame indices to sample (evenly distributed)
            if total_frames <= num_frames:
                frame_indices = list(range(total_frames))
            else:
                frame_indices = [int(i * (total_frames / num_frames)) for i in range(num_frames)]
            
            frames = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                    
            cap.release()
            return frames
            
        except Exception as e:
            logger.error(f"Error extracting frames from {video_path}: {e}")
            return []
    
    def compute_visual_similarity(self, original_frames: List[np.ndarray], 
                                 washed_frames: List[np.ndarray]) -> float:
        """
        Compute visual similarity between original and washed frames.
        
        Args:
            original_frames: List of original video frames
            washed_frames: List of corresponding washed video frames
            
        Returns:
            Similarity score between 0.0 and 1.0 (higher is more similar)
        """
        if not original_frames or not washed_frames:
            return 0.0
            
        # Ensure equal number of frames to compare
        num_frames = min(len(original_frames), len(washed_frames))
        
        # Calculate structural similarity index (SSIM) for each frame pair
        similarities = []
        for i in range(num_frames):
            # Convert to grayscale for comparison
            orig_gray = cv2.cvtColor(original_frames[i], cv2.COLOR_BGR2GRAY)
            wash_gray = cv2.cvtColor(washed_frames[i], cv2.COLOR_BGR2GRAY)
            
            # Calculate SSIM
            try:
                # Use a simple MSE-based similarity if SSIM is not available
                mse = np.mean((orig_gray - wash_gray) ** 2)
                similarity = 1 - (mse / 255**2)  # Convert to 0-1 range
                similarities.append(similarity)
            except Exception as e:
                logger.warning(f"Error computing similarity: {e}")
                continue
                
        # Return average similarity
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def compute_hash_difference(self, original_frames: List[np.ndarray], 
                               washed_frames: List[np.ndarray]) -> float:
        """
        Compute perceptual hash difference between original and washed frames.
        
        Args:
            original_frames: List of original video frames
            washed_frames: List of corresponding washed video frames
            
        Returns:
            Hash difference score between 0.0 and 1.0 (higher means more different)
        """
        if not original_frames or not washed_frames:
            return 0.0
            
        # Ensure equal number of frames to compare
        num_frames = min(len(original_frames), len(washed_frames))
        
        # Calculate perceptual hash difference for each frame pair
        hash_diffs = []
        for i in range(num_frames):
            # Convert OpenCV BGR to PIL RGB for imagehash
            orig_pil = Image.fromarray(cv2.cvtColor(original_frames[i], cv2.COLOR_BGR2RGB))
            wash_pil = Image.fromarray(cv2.cvtColor(washed_frames[i], cv2.COLOR_BGR2RGB))
            
            # Compute perceptual hash
            orig_hash = imagehash.phash(orig_pil)
            wash_hash = imagehash.phash(wash_pil)
            
            # Calculate hash difference (normalized to 0-1)
            hash_diff = (orig_hash - wash_hash) / 64.0  # phash is 64 bits
            hash_diffs.append(hash_diff)
                
        # Return average hash difference
        return sum(hash_diffs) / len(hash_diffs) if hash_diffs else 0.0
    
    def extract_metadata(self, video_path: str) -> Dict[str, str]:
        """
        Extract metadata from the video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary of metadata key-value pairs
        """
        try:
            # Use ffprobe to extract metadata
            cmd = [
                'ffprobe', 
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning(f"Error extracting metadata: {result.stderr}")
                return {}
                
            import json
            metadata = json.loads(result.stdout)
            
            # Extract relevant metadata fields
            result = {}
            if 'format' in metadata and 'tags' in metadata['format']:
                result.update(metadata['format']['tags'])
                
            # Add basic format info
            if 'format' in metadata:
                for key in ['format_name', 'duration', 'bit_rate']:
                    if key in metadata['format']:
                        result[key] = metadata['format'][key]
            
            # Add video stream info
            if 'streams' in metadata:
                for stream in metadata['streams']:
                    if stream.get('codec_type') == 'video':
                        for key in ['codec_name', 'width', 'height', 'r_frame_rate']:
                            if key in stream:
                                result[f"video_{key}"] = stream[key]
                        if 'tags' in stream:
                            for tag_key, tag_value in stream['tags'].items():
                                result[f"video_tag_{tag_key}"] = tag_value
                        break
                        
            return result
            
        except Exception as e:
            logger.error(f"Error extracting metadata from {video_path}: {e}")
            return {}
    
    def compute_metadata_difference(self, orig_metadata: Dict[str, str], 
                                   washed_metadata: Dict[str, str]) -> float:
        """
        Compute difference between original and washed video metadata.
        
        Args:
            orig_metadata: Original video metadata
            washed_metadata: Washed video metadata
            
        Returns:
            Metadata difference score between 0.0 and 1.0 (higher means more different)
        """
        # Get all unique keys
        all_keys = set(orig_metadata.keys()) | set(washed_metadata.keys())
        
        if not all_keys:
            return 0.0
            
        # Count differences
        differences = 0
        for key in all_keys:
            if key not in orig_metadata or key not in washed_metadata:
                differences += 1
            elif orig_metadata[key] != washed_metadata[key]:
                differences += 1
                
        # Return normalized difference score
        return differences / len(all_keys)
    
    def verify_washing_effectiveness(self, original_video: str, washed_video: str) -> Dict[str, float]:
        """
        Verify how effectively a video has been washed.
        
        Args:
            original_video: Path to the original video
            washed_video: Path to the washed video
            
        Returns:
            Dictionary with effectiveness metrics:
            - visual_similarity: How visually similar the videos are (0.0-1.0)
            - hash_difference: How different the perceptual hashes are (0.0-1.0)
            - metadata_difference: How different the metadata is (0.0-1.0)
            - overall_effectiveness: Combined score (0.0-1.0)
        """
        # Extract frames for comparison
        orig_frames = self.extract_frames(original_video)
        washed_frames = self.extract_frames(washed_video)
        
        # Compute visual similarity
        visual_similarity = self.compute_visual_similarity(orig_frames, washed_frames)
        
        # Compute hash difference
        hash_difference = self.compute_hash_difference(orig_frames, washed_frames)
        
        # Extract and compare metadata
        orig_metadata = self.extract_metadata(original_video)
        washed_metadata = self.extract_metadata(washed_video)
        metadata_difference = self.compute_metadata_difference(orig_metadata, washed_metadata)
        
        # Compute overall effectiveness score
        # We want high visual similarity but high hash and metadata differences
        # Formula: hash_diff * metadata_diff * visual_similarity
        overall_effectiveness = hash_difference * metadata_difference * visual_similarity
        
        return {
            'visual_similarity': visual_similarity,
            'hash_difference': hash_difference,
            'metadata_difference': metadata_difference,
            'overall_effectiveness': overall_effectiveness
        }
    
    def get_improvement_suggestions(self, metrics: Dict[str, float]) -> List[str]:
        """
        Generate suggestions for improving washing effectiveness based on metrics.
        
        Args:
            metrics: Dictionary with effectiveness metrics from verify_washing_effectiveness
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        # Visual similarity too low
        if metrics['visual_similarity'] < 0.9:
            suggestions.append("Reduce visual effect intensity to maintain better appearance")
            
        # Hash difference too low
        if metrics['hash_difference'] < 0.4:
            suggestions.append("Increase pixel-level noise or subtle visual distortions")
            
        # Metadata difference too low
        if metrics['metadata_difference'] < 0.5:
            suggestions.append("Add more metadata variation (timestamps, device info)")
            
        # Overall effectiveness too low
        if metrics['overall_effectiveness'] < 0.3:
            suggestions.append("Try combining multiple washing techniques")
            
        return suggestions 