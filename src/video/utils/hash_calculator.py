import cv2
import numpy as np
import hashlib
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union
from ..utils.logging_utils import configure_logger, timed, log_exceptions

# Configure logger
logger = configure_logger("HashCalculator")

class ConsistentVideoHasher:
    """
    A class for consistently calculating perceptual hashes of videos.
    Uses position-based frame sampling and multiple hash algorithms
    to ensure consistency across runs.
    """
    
    def __init__(self, 
                 num_sample_frames=10, 
                 resize_dim=(256, 256),
                 hash_size=8):
        """
        Initialize the video hasher.
        
        Args:
            num_sample_frames: Number of frames to sample from the video
            resize_dim: Dimensions to resize frames to before hashing
            hash_size: Size of the hash grid (hash_size x hash_size)
        """
        self.num_sample_frames = num_sample_frames
        self.resize_dim = resize_dim
        self.hash_size = hash_size
        self.interpolation = cv2.INTER_LANCZOS4  # Use consistent interpolation
        
    @timed
    @log_exceptions
    def calculate_hash(self, video_path):
        """
        Calculate consistent hash for a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary with hash information
        """
        start_time = time.time()
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        logger.info(f"Calculating hash for: {video_path}")
        
        # Open the video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video properties: {frame_count} frames, {fps:.2f} FPS, {width}x{height}")
        
        # Calculate fixed frame positions
        frame_positions = self._calculate_frame_positions(frame_count)
        logger.info(f"Using {len(frame_positions)} fixed frame positions: {frame_positions}")
        
        # Process frames at fixed positions
        frame_hashes = {}
        for pos in frame_positions:
            # Set position and read frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            
            if not ret:
                logger.warning(f"Failed to read frame at position {pos}")
                continue
                
            # Calculate multiple hash types for the frame
            hashes = self._calculate_frame_hashes(frame)
            frame_hashes[pos] = hashes
            
        cap.release()
        
        # Generate consistent video hash
        video_hash = self._generate_video_hash(frame_hashes, frame_positions, video_path, 
                                              frame_count, fps)
        
        execution_time = time.time() - start_time
        logger.info(f"Hash calculation completed in {execution_time:.2f}s")
        
        return video_hash
    
    def _calculate_frame_positions(self, frame_count):
        """
        Calculate fixed frame positions to sample from the video.
        
        Args:
            frame_count: Total number of frames in the video
            
        Returns:
            List of frame positions to sample
        """
        if frame_count <= self.num_sample_frames:
            # If video has fewer frames than requested samples, use all frames
            return list(range(frame_count))
        
        # Calculate evenly distributed frame positions
        positions = []
        for i in range(self.num_sample_frames):
            # This formula ensures we include the first and last frames
            # and distribute the rest evenly
            pos = int(i * (frame_count - 1) / (self.num_sample_frames - 1))
            positions.append(pos)
            
        return positions
    
    def _calculate_frame_hashes(self, frame):
        """
        Calculate multiple hash types for a single frame.
        
        Args:
            frame: Video frame as numpy array
            
        Returns:
            Dictionary with different hash types
        """
        # Resize the frame to standard dimensions
        resized_frame = cv2.resize(frame, self.resize_dim, interpolation=self.interpolation)
        
        # Convert to grayscale
        gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate different hash types
        average_hash = self._average_hash(gray)
        perceptual_hash = self._perceptual_hash(gray)
        difference_hash = self._difference_hash(gray)
        
        return {
            'average': average_hash,
            'perceptual': perceptual_hash,
            'difference': difference_hash
        }
    
    def _average_hash(self, gray_img):
        """
        Calculate average hash (aHash) for an image.
        
        Args:
            gray_img: Grayscale image
            
        Returns:
            Hexadecimal hash string
        """
        # Resize to hash size
        resized = cv2.resize(gray_img, (self.hash_size, self.hash_size), 
                            interpolation=self.interpolation)
        
        # Calculate average pixel value
        avg_pixel = resized.mean()
        
        # Create binary hash: 1 for pixels above average, 0 for pixels below
        hash_bits = resized >= avg_pixel
        
        # Convert to hex string
        hash_str = ''.join(['1' if bit else '0' for bit in hash_bits.flatten()])
        hash_hex = hex(int(hash_str, 2))[2:].zfill(self.hash_size * self.hash_size // 4)
        
        return hash_hex
    
    def _perceptual_hash(self, gray_img):
        """
        Calculate perceptual hash (pHash) using DCT for an image.
        
        Args:
            gray_img: Grayscale image
            
        Returns:
            Hexadecimal hash string
        """
        # Resize to 4x hash size (DCT works better with more pixels)
        img_size = self.hash_size * 4
        resized = cv2.resize(gray_img, (img_size, img_size), 
                            interpolation=self.interpolation)
        
        # Convert to float and calculate DCT
        dct_img = cv2.dct(np.float32(resized))
        
        # Keep only the low-frequency components
        dct_low = dct_img[:self.hash_size, :self.hash_size]
        
        # Use the median as threshold (more robust than mean)
        dct_median = np.median(dct_low)
        
        # Create binary hash
        hash_bits = dct_low >= dct_median
        
        # Convert to hex string
        hash_str = ''.join(['1' if bit else '0' for bit in hash_bits.flatten()])
        hash_hex = hex(int(hash_str, 2))[2:].zfill(self.hash_size * self.hash_size // 4)
        
        return hash_hex
    
    def _difference_hash(self, gray_img):
        """
        Calculate difference hash (dHash) for an image.
        
        Args:
            gray_img: Grayscale image
            
        Returns:
            Hexadecimal hash string
        """
        # Resize to hash_size+1 x hash_size (we need an extra column for differences)
        resized = cv2.resize(gray_img, (self.hash_size + 1, self.hash_size), 
                            interpolation=self.interpolation)
        
        # Calculate differences between adjacent pixels
        diff = resized[:, 1:] >= resized[:, :-1]
        
        # Convert to hex string
        hash_str = ''.join(['1' if bit else '0' for bit in diff.flatten()])
        hash_hex = hex(int(hash_str, 2))[2:].zfill(self.hash_size * self.hash_size // 4)
        
        return hash_hex
    
    def _generate_video_hash(self, frame_hashes, frame_positions, video_path, 
                            frame_count, fps):
        """
        Generate a consistent hash for the entire video.
        
        Args:
            frame_hashes: Dictionary of frame hashes
            frame_positions: List of frame positions
            video_path: Path to the video file
            frame_count: Total number of frames
            fps: Frames per second
            
        Returns:
            Dictionary with hash information
        """
        # Create positional hash string (frame position -> hash)
        pos_hashes = []
        for pos in sorted(frame_hashes.keys()):
            hashes = frame_hashes[pos]
            # Combine all hash types for this position
            pos_hash = f"{pos}:{hashes['average']}:{hashes['perceptual']}:{hashes['difference']}"
            pos_hashes.append(pos_hash)
        
        # Join all position+hash strings
        combined_hash_str = "_".join(pos_hashes)
        
        # Add metadata to make the hash even more consistent
        file_size = Path(video_path).stat().st_size
        metadata = f"{frame_count}:{fps:.2f}:{file_size}:{len(frame_positions)}"
        
        # Create final hash strings
        hash_input = f"{metadata}_{combined_hash_str}"
        main_hash = hashlib.md5(hash_input.encode()).hexdigest()
        
        # Create separate hashes for each algorithm
        avg_hash_input = "_".join([f"{pos}:{frame_hashes[pos]['average']}" 
                                  for pos in sorted(frame_hashes.keys())])
        perceptual_hash_input = "_".join([f"{pos}:{frame_hashes[pos]['perceptual']}" 
                                         for pos in sorted(frame_hashes.keys())])
        diff_hash_input = "_".join([f"{pos}:{frame_hashes[pos]['difference']}" 
                                   for pos in sorted(frame_hashes.keys())])
        
        avg_hash = hashlib.md5(avg_hash_input.encode()).hexdigest()
        perceptual_hash = hashlib.md5(perceptual_hash_input.encode()).hexdigest()
        diff_hash = hashlib.md5(diff_hash_input.encode()).hexdigest()
        
        # Return all hash information
        return {
            'hash': main_hash,
            'avg_hash': avg_hash,
            'perceptual_hash': perceptual_hash,
            'diff_hash': diff_hash,
            'frame_count': frame_count,
            'fps': fps,
            'file_size': file_size,
            'sampled_frames': len(frame_positions),
            'frame_positions': frame_positions,
            'version': '1.0'  # Version for future compatibility
        }
    
    @timed
    @log_exceptions
    def compare_hashes(self, hash1, hash2, verbose=True):
        """
        Compare two video hashes and calculate similarity.
        
        Args:
            hash1: First hash dictionary
            hash2: Second hash dictionary
            verbose: Whether to print details
            
        Returns:
            Dictionary with comparison results
        """
        if hash1.get('version') != hash2.get('version'):
            logger.warning("Comparing hashes from different versions")
        
        # Calculate hash differences
        main_diff = self._calculate_hash_difference(hash1['hash'], hash2['hash'])
        avg_diff = self._calculate_hash_difference(hash1['avg_hash'], hash2['avg_hash'])
        perceptual_diff = self._calculate_hash_difference(hash1['perceptual_hash'], 
                                                         hash2['perceptual_hash'])
        diff_diff = self._calculate_hash_difference(hash1['diff_hash'], hash2['diff_hash'])
        
        # Calculate weighted similarity (0-100%, higher is more similar)
        similarity = 100 - (main_diff * 0.4 + avg_diff * 0.2 + 
                           perceptual_diff * 0.2 + diff_diff * 0.2)
        
        if verbose:
            logger.info(f"Main hash difference: {main_diff:.2f}%")
            logger.info(f"Average hash difference: {avg_diff:.2f}%")
            logger.info(f"Perceptual hash difference: {perceptual_diff:.2f}%")
            logger.info(f"Difference hash difference: {diff_diff:.2f}%")
            logger.info(f"Overall similarity: {similarity:.2f}%")
            
            # Check metadata
            if hash1['frame_count'] != hash2['frame_count']:
                logger.info(f"Frame count differs: {hash1['frame_count']} vs {hash2['frame_count']}")
            if abs(hash1['fps'] - hash2['fps']) > 0.01:
                logger.info(f"FPS differs: {hash1['fps']:.2f} vs {hash2['fps']:.2f}")
            if hash1['file_size'] != hash2['file_size']:
                logger.info(f"File size differs: {hash1['file_size']} vs {hash2['file_size']}")
        
        return {
            'similarity': similarity,
            'main_diff': main_diff,
            'avg_diff': avg_diff,
            'perceptual_diff': perceptual_diff,
            'diff_diff': diff_diff,
            'is_same_video': similarity >= 95.0  # Threshold for considering same video
        }
    
    def _calculate_hash_difference(self, hash1, hash2):
        """
        Calculate the percentage difference between two hash strings.
        
        Args:
            hash1, hash2: Hash strings to compare
            
        Returns:
            Percentage difference (0-100%)
        """
        if hash1 == hash2:
            return 0.0
            
        # For MD5 hashes, we convert to binary and calculate bit difference
        h1_bin = bin(int(hash1, 16))[2:].zfill(len(hash1) * 4)
        h2_bin = bin(int(hash2, 16))[2:].zfill(len(hash2) * 4)
        
        # Calculate Hamming distance (number of differing bits)
        distance = sum(b1 != b2 for b1, b2 in zip(h1_bin, h2_bin))
        max_distance = len(h1_bin)
        
        # Return as percentage
        return (distance / max_distance) * 100

# Create a default instance for convenience
default_hasher = ConsistentVideoHasher(
    num_sample_frames=15,  # Number of frames to sample
    resize_dim=(256, 256),  # Resize dimensions
    hash_size=16  # Hash grid size (16x16)
)

# Convenience functions that use the default hasher
@timed
@log_exceptions
def calculate_video_hash(video_path: str) -> Dict:
    """Calculate hash for a video using the default hasher."""
    return default_hasher.calculate_hash(video_path)

@timed
@log_exceptions
def calculate_video_difference(video1_path: str, video2_path: str) -> float:
    """Calculate difference between two videos using the default hasher."""
    hash1 = default_hasher.calculate_hash(video1_path)
    hash2 = default_hasher.calculate_hash(video2_path)
    comparison = default_hasher.compare_hashes(hash1, hash2, verbose=False)
    return comparison['main_diff']  # Return just the main hash difference 