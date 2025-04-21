import logging
import subprocess
import os
from typing import Tuple, Dict, Any, Optional

# Try importing GPU libraries with graceful fallbacks
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

from ..utils.logging_utils import configure_logger

# Configure logger
logger = configure_logger("GPUManager")

class GPUManager:
    """
    Utility class to manage GPU resources, check availability,
    and provide fallback mechanisms for video processing.
    """
    
    def __init__(self, memory_threshold: float = 0.8, force_cpu: bool = False):
        """
        Initialize the GPU manager.
        
        Args:
            memory_threshold: Threshold (0.0-1.0) for GPU memory usage before falling back to CPU
            force_cpu: If True, always use CPU regardless of GPU availability
        """
        self.memory_threshold = max(0.1, min(0.95, memory_threshold))
        self.force_cpu = force_cpu
        
        # Check GPU availability
        self.has_torch = TORCH_AVAILABLE
        self.has_cupy = CUPY_AVAILABLE
        self.has_cuda = self._check_cuda_available() and not force_cpu
        self.has_nvenc = self._check_nvenc_available() and not force_cpu
        
        # Initialize device
        self.device = self._get_device()
        
        # Log GPU status
        self._log_gpu_status()
    
    def _check_cuda_available(self) -> bool:
        """Check if CUDA is available through PyTorch."""
        if not TORCH_AVAILABLE:
            return False
        return torch.cuda.is_available()
    
    def _check_nvenc_available(self) -> bool:
        """Check if NVENC is available for hardware encoding."""
        try:
            result = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], 
                               capture_output=True, text=True, timeout=5)
            return 'h264_nvenc' in result.stdout
        except Exception as e:
            logger.warning(f"Error checking NVENC availability: {e}")
            return False
    
    def _get_device(self) -> Any:
        """Get the appropriate torch device."""
        if not self.has_torch:
            return None
        
        if self.has_cuda:
            return torch.device('cuda')
        return torch.device('cpu')
    
    def _log_gpu_status(self):
        """Log GPU availability and stats."""
        if self.force_cpu:
            logger.info("Force CPU mode enabled, ignoring GPU")
            return
            
        if self.has_cuda:
            gpu_name = torch.cuda.get_device_name(0)
            total_memory_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            logger.info(f"GPU available: {gpu_name} with {total_memory_mb:.0f} MB memory")
            
            if self.has_nvenc:
                logger.info("NVENC hardware encoder available")
            else:
                logger.warning("NVENC hardware encoder not available")
        else:
            logger.warning("CUDA not available, using CPU for processing")
    
    def should_use_gpu(self, estimated_memory_mb: float = 0) -> bool:
        """
        Determine if operation should use GPU based on current memory state.
        
        Args:
            estimated_memory_mb: Estimated memory requirement in MB
            
        Returns:
            True if GPU should be used, False otherwise
        """
        if not self.has_cuda or self.force_cpu:
            return False
            
        # Check current GPU memory usage
        current_usage = self.get_memory_usage()
        
        # Get total memory
        total_memory_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        
        # Calculate if we have enough free memory
        free_memory_mb = total_memory_mb * (1 - current_usage)
        
        # Add a safety buffer (20% of requested memory)
        required_memory = estimated_memory_mb * 1.2
        
        can_use_gpu = free_memory_mb >= required_memory
        
        if not can_use_gpu and estimated_memory_mb > 0:
            logger.warning(f"Insufficient GPU memory for operation. Need {required_memory:.0f}MB, free: {free_memory_mb:.0f}MB")
            
        return can_use_gpu
    
    def get_memory_usage(self) -> float:
        """
        Get current GPU memory usage as a fraction (0.0-1.0).
        
        Returns:
            Memory usage ratio
        """
        if not self.has_cuda:
            return 0.0
            
        try:
            allocated = torch.cuda.memory_allocated()
            max_allocated = torch.cuda.max_memory_allocated()
            total_memory = torch.cuda.get_device_properties(0).total_memory
            
            # Return current usage against total memory
            return allocated / total_memory
        except Exception as e:
            logger.error(f"Error checking GPU memory: {e}")
            return 1.0  # Assume full for safety
    
    def cleanup(self):
        """Force GPU memory cleanup."""
        if self.has_cuda:
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.debug("GPU memory cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up GPU memory: {e}")
    
    def estimate_frame_memory(self, height: int, width: int, batch_size: int = 1) -> float:
        """
        Estimate memory required for frame processing.
        
        Args:
            height: Frame height in pixels
            width: Frame width in pixels
            batch_size: Number of frames to process in a batch
            
        Returns:
            Estimated memory in MB
        """
        # Rough estimate: 4 bytes per pixel (float32) * 3 channels * dimensions * batch
        bytes_per_frame = 4 * 3 * height * width
        
        # Account for intermediate buffers (2x for safety)
        memory_mb = (bytes_per_frame * batch_size * 2) / (1024 * 1024)
        
        return memory_mb
    
    def get_optimal_batch_size(self, frame_height: int, frame_width: int) -> int:
        """
        Calculate optimal batch size for frame processing based on GPU memory.
        
        Args:
            frame_height: Frame height in pixels
            frame_width: Frame width in pixels
            
        Returns:
            Optimal batch size
        """
        if not self.has_cuda or self.force_cpu:
            return 1  # No batching on CPU
            
        # Get free memory
        total_memory_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        current_usage = self.get_memory_usage()
        free_memory_mb = total_memory_mb * (1 - current_usage) * self.memory_threshold
        
        # Calculate memory per frame
        memory_per_frame_mb = self.estimate_frame_memory(frame_height, frame_width, 1)
        
        # Calculate batch size (min 1, max 64)
        batch_size = max(1, min(64, int(free_memory_mb / memory_per_frame_mb)))
        
        logger.debug(f"Optimal batch size: {batch_size} for {frame_height}x{frame_width} frames")
        return batch_size
    
    def get_nvenc_params(self, quality: str = 'medium') -> Dict[str, Any]:
        """
        Get optimal NVENC parameters based on quality setting.
        
        Args:
            quality: Quality level ('low', 'medium', 'high')
            
        Returns:
            Dictionary of NVENC parameters
        """
        if not self.has_nvenc:
            return {}
            
        # Base parameters
        params = {
            'encoder': 'h264_nvenc',
            'use_gpu': True
        }
        
        # Quality-specific parameters
        if quality == 'low':
            params.update({
                'preset': 'p1',  # Fastest preset
                'tune': 'fastdecode',
                'rc': 'vbr',
                'cq': 28
            })
        elif quality == 'high':
            params.update({
                'preset': 'p7',  # High quality preset
                'tune': 'hq',
                'rc': 'vbr_hq',
                'cq': 15
            })
        else:  # medium (default)
            params.update({
                'preset': 'p4',  # Balanced preset
                'tune': 'hq',
                'rc': 'vbr',
                'cq': 21
            })
            
        return params
    
    def get_ffmpeg_gpu_args(self, quality: str = 'medium') -> list:
        """
        Get FFmpeg command-line arguments for GPU encoding.
        
        Args:
            quality: Quality level ('low', 'medium', 'high')
            
        Returns:
            List of FFmpeg command-line arguments
        """
        if not self.has_nvenc:
            logger.warning("NVENC not available for hardware encoding, falling back to CPU")
            return []
            
        nvenc_params = self.get_nvenc_params(quality)
        
        # Base NVENC arguments
        args = [
            '-c:v', 'h264_nvenc',
            '-gpu', '0',
        ]
        
        # Add quality-specific parameters
        if quality == 'low':
            args.extend([
                '-preset', 'fast',
                '-tune', 'hq',  # Changed from fastdecode to preserve colors
                '-rc', 'vbr',
                '-cq', '26',   # Changed from 28 for better quality
                '-b:v', '5M'
            ])
        elif quality == 'high':
            args.extend([
                '-preset', 'slow',
                '-tune', 'hq',
                '-rc', 'vbr_hq',
                '-cq', '15',
                '-b:v', '15M'  # Increased from 12M for better quality
            ])
        else:  # medium (default)
            args.extend([
                '-preset', 'medium',
                '-tune', 'hq',
                '-rc', 'vbr',
                '-cq', '18',   # Changed from 21 for better quality
                '-b:v', '10M'  # Increased from 8M for better quality
            ])
        
        # Add additional parameters for better GPU utilization and color preservation
        args.extend([
            '-profile:v', 'high',        # High profile for better quality
            '-spatial_aq', '1',          # Spatial adaptive quantization
            '-temporal_aq', '1',         # Temporal adaptive quantization
            '-refs', '3',                # Reference frames for better compression
            '-bf', '3',                  # B-frames for better compression
            '-g', '250',                 # Keyframe interval
            '-pix_fmt', 'yuv420p',       # Pixel format for better compatibility
            '-color_primaries', 'bt709', # Preserve color primaries
            '-color_trc', 'bt709',       # Preserve color transfer characteristics
            '-colorspace', 'bt709',      # Preserve color space
            '-strict', 'normal'          # More strict color handling
        ])
        
        logger.info(f"Using NVENC GPU encoding with quality: {quality}")
        return args

# Create a global instance for easy import
gpu_manager = GPUManager() 