import logging
import numpy as np
from typing import Optional, Union, Tuple, Callable, Any
import random

# Try importing GPU libraries with graceful fallbacks
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..utils.logging_utils import configure_logger, timed, log_exceptions
from ..utils.gpu_manager import gpu_manager

# Configure logger
logger = configure_logger("GPUEffects")

# Helper functions to handle tensor conversion
def _frame_to_tensor(frame: np.ndarray) -> torch.Tensor:
    """Convert a numpy frame to a PyTorch tensor on the appropriate device."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for GPU effects")
        
    if isinstance(frame, torch.Tensor):
        return frame.to(gpu_manager.device)
        
    # Store original data - we'll keep as uint8 until specific operations need float
    # This preserves exact colors and is more memory-efficient
    tensor = torch.from_numpy(frame.copy()).to(gpu_manager.device)
    
    # Handle RGB or BGR input and convert to channels-first format
    if tensor.dim() == 3 and tensor.shape[2] in (3, 4):  # (H,W,C)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (1,C,H,W)
    elif tensor.dim() == 4 and tensor.shape[3] in (3, 4):  # (B,H,W,C)
        tensor = tensor.permute(0, 3, 1, 2)  # (B,C,H,W)
    
    return tensor

def _tensor_to_frame(tensor: torch.Tensor) -> np.ndarray:
    """Convert a PyTorch tensor back to numpy frame."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for GPU effects")
        
    if not isinstance(tensor, torch.Tensor):
        return tensor
    
    # Move to CPU
    tensor = tensor.detach().cpu()
    
    # Handle different tensor formats
    if tensor.dim() == 4:  # (B,C,H,W)
        if tensor.shape[0] == 1:  # Single frame
            tensor = tensor.squeeze(0)
            # (C,H,W) -> (H,W,C)
            tensor = tensor.permute(1, 2, 0)
        else:
            # (B,C,H,W) -> (B,H,W,C)
            tensor = tensor.permute(0, 2, 3, 1)
    
    # Convert to uint8 for image without normalization changes
    if tensor.dtype != torch.uint8:
        tensor = torch.clamp(tensor, 0, 255).to(torch.uint8)
        
    return tensor.numpy()

def _apply_effect(frame: np.ndarray, effect_fn: Callable, use_gpu: bool = True, **kwargs) -> np.ndarray:
    """
    Apply an effect function to a frame, handling GPU/CPU dispatch.
    
    Args:
        frame: Input video frame
        effect_fn: Function to apply for the effect
        use_gpu: Whether to use GPU acceleration
        **kwargs: Additional parameters for the effect
        
    Returns:
        Processed frame
    """
    if use_gpu and gpu_manager.should_use_gpu():
        # Convert to tensor and apply effect
        tensor = _frame_to_tensor(frame)
        result_tensor = effect_fn(tensor, **kwargs)
        return _tensor_to_frame(result_tensor)
    else:
        # Fall back to CPU implementation 
        # This needs to be implemented for each effect
        if 'cpu_fn' in kwargs:
            return kwargs['cpu_fn'](frame, **kwargs)
        return frame

# =====================
# GPU Effect Functions 
# =====================

@timed
@log_exceptions
def zoom_frame_gpu(tensor: torch.Tensor, zoom_factor: float = 1.2, **kwargs) -> torch.Tensor:
    """
    Apply zoom effect to a frame tensor.
    
    Args:
        tensor: Input tensor in (B,C,H,W) format
        zoom_factor: Zoom factor (>1 for zoom in, <1 for zoom out)
        **kwargs: Additional parameters (including cpu_fn for fallback)
        
    Returns:
        Zoomed tensor
    """
    # Get dimensions
    if tensor.dim() == 3:  # (C,H,W)
        tensor = tensor.unsqueeze(0)  # Add batch dimension
    
    # Store original dtype for later conversion back
    original_dtype = tensor.dtype
    
    # Convert to float for interpolation
    if original_dtype != torch.float32:
        tensor = tensor.float()
    
    b, c, h, w = tensor.shape
    
    # Calculate new dimensions
    new_h = int(h * zoom_factor)
    new_w = int(w * zoom_factor)
    
    # Resize using bilinear interpolation
    zoomed = F.interpolate(
        tensor, 
        size=(new_h, new_w), 
        mode='bilinear',
        align_corners=False
    )
    
    # Crop to original size from center
    start_h = (new_h - h) // 2
    start_w = (new_w - w) // 2
    cropped = zoomed[:, :, start_h:start_h+h, start_w:start_w+w]
    
    # Convert back to original dtype if needed
    if original_dtype != torch.float32:
        cropped = torch.clamp(cropped, 0, 255).to(original_dtype)
    
    return cropped

@timed
@log_exceptions
def crop_frame_gpu(tensor: torch.Tensor, crop_percent: float = 0.1, **kwargs) -> torch.Tensor:
    """
    Apply crop effect to a frame tensor.
    
    Args:
        tensor: Input tensor in (B,C,H,W) format
        crop_percent: Amount to crop from edges (0.0-1.0)
        **kwargs: Additional parameters (including cpu_fn for fallback)
        
    Returns:
        Cropped tensor
    """
    # Get dimensions
    if tensor.dim() == 3:  # (C,H,W)
        tensor = tensor.unsqueeze(0)  # Add batch dimension
    
    # Store original dtype for later conversion back
    original_dtype = tensor.dtype
    
    # Convert to float for interpolation
    if original_dtype != torch.float32:
        tensor = tensor.float()
    
    b, c, h, w = tensor.shape
    
    # Calculate crop boundaries
    crop_h = int(h * crop_percent)
    crop_w = int(w * crop_percent)
    
    # Crop the image
    cropped = tensor[:, :, crop_h:h-crop_h, crop_w:w-crop_w]
    
    # Resize back to original dimensions
    result = F.interpolate(
        cropped, 
        size=(h, w), 
        mode='bilinear',
        align_corners=False
    )
    
    # Convert back to original dtype if needed
    if original_dtype != torch.float32:
        result = torch.clamp(result, 0, 255).to(original_dtype)
    
    return result

@timed
@log_exceptions
def filter_frame_gpu(tensor: torch.Tensor, filter_type: str = 'brightness', intensity: float = 0.2, **kwargs) -> torch.Tensor:
    """
    Apply various filters to a frame tensor.
    
    Args:
        tensor: Input tensor in (B,C,H,W) format
        filter_type: Type of filter ('brightness', 'contrast', 'saturation', etc.)
        intensity: Effect intensity (0.0-1.0)
        **kwargs: Additional parameters (including cpu_fn for fallback)
        
    Returns:
        Filtered tensor
    """
    # Get dimensions
    if tensor.dim() == 3:  # (C,H,W)
        tensor = tensor.unsqueeze(0)  # Add batch dimension
    
    # Store original dtype for later conversion back
    original_dtype = tensor.dtype
    
    # Make a copy of the original tensor and convert to float for processing
    tensor_float = tensor.float() / 255.0
    
    # Apply requested filter
    if filter_type == 'brightness':
        # Adjust brightness by adding/subtracting
        factor = intensity * 2 - 1  # Map 0-1 to -1 to 1
        result = tensor_float + factor
        
    elif filter_type == 'contrast':
        # Adjust contrast
        factor = 1 + intensity  # 1.0 - 2.0
        mean = torch.mean(tensor_float, dim=[2, 3], keepdim=True)
        result = (tensor_float - mean) * factor + mean
        
    elif filter_type == 'saturation':
        # Convert to grayscale
        # RGB weights for luminance: [0.299, 0.587, 0.114]
        rgb_weights = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1).to(tensor.device)
        grayscale = torch.sum(tensor_float * rgb_weights, dim=1, keepdim=True)
        grayscale = grayscale.expand_as(tensor_float)
        
        # Blend with original based on intensity
        factor = 1 + intensity  # 1.0 - 2.0
        result = torch.lerp(grayscale, tensor_float, factor)
        
    elif filter_type == 'hue':
        # For simple hue adjustment, we'll use a color rotation matrix
        # This is a simplification of HSV hue rotation
        angle = intensity * 2 * 3.14159  # Convert intensity to angle in radians
        
        # Color rotation matrix
        cos_val = torch.cos(torch.tensor(angle))
        sin_val = torch.sin(torch.tensor(angle))
        
        rot_matrix = torch.tensor([
            [0.299 + 0.701 * cos_val, 0.587 - 0.587 * cos_val + 0.114 * sin_val, 0.114 - 0.114 * cos_val - 0.587 * sin_val],
            [0.299 - 0.299 * cos_val - 0.114 * sin_val, 0.587 + 0.413 * cos_val, 0.114 - 0.114 * cos_val + 0.299 * sin_val],
            [0.299 - 0.299 * cos_val + 0.587 * sin_val, 0.587 - 0.587 * cos_val - 0.299 * sin_val, 0.114 + 0.886 * cos_val]
        ]).to(tensor.device)
        
        # Apply rotation to each pixel
        b, c, h, w = tensor_float.shape
        pixels = tensor_float.permute(0, 2, 3, 1).reshape(-1, 3)  # (b*h*w, 3)
        rotated = torch.matmul(pixels, rot_matrix.T)
        result = rotated.reshape(b, h, w, 3).permute(0, 3, 1, 2)  # (b, 3, h, w)
    else:
        logger.warning(f"Unknown filter type: {filter_type}, using original")
        result = tensor_float
    
    # Clamp to valid range and convert back to original dtype
    result = torch.clamp(result * 255.0, 0, 255)
    
    # Convert back to original dtype if needed
    if original_dtype != torch.float32:
        result = result.to(original_dtype)
    
    return result

@timed
@log_exceptions
def blur_frame_gpu(tensor: torch.Tensor, radius: int = 5, **kwargs) -> torch.Tensor:
    """
    Apply Gaussian blur to a frame tensor.
    
    Args:
        tensor: Input tensor in (B,C,H,W) format
        radius: Blur radius
        **kwargs: Additional parameters (including cpu_fn for fallback)
        
    Returns:
        Blurred tensor
    """
    # Get dimensions
    if tensor.dim() == 3:  # (C,H,W)
        tensor = tensor.unsqueeze(0)  # Add batch dimension
    
    # Ensure radius is odd
    kernel_size = max(3, radius * 2 + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Calculate sigma
    sigma = radius / 3.0
    
    # Apply gaussian blur
    result = tensor.float()  # Ensure float for blurring
    
    # Apply separable Gaussian blur for efficiency
    padding = kernel_size // 2
    
    # First blur horizontally
    result = F.pad(result, (padding, padding, 0, 0), mode='reflect')
    result = F.conv2d(
        result,
        _gaussian_kernel_1d(kernel_size, sigma).view(1, 1, 1, kernel_size).expand(
            result.shape[1], 1, 1, kernel_size
        ).to(tensor.device),
        groups=result.shape[1]
    )
    
    # Then blur vertically
    result = F.pad(result, (0, 0, padding, padding), mode='reflect')
    result = F.conv2d(
        result,
        _gaussian_kernel_1d(kernel_size, sigma).view(1, 1, kernel_size, 1).expand(
            result.shape[1], 1, kernel_size, 1
        ).to(tensor.device),
        groups=result.shape[1]
    )
    
    return result

def _gaussian_kernel_1d(kernel_size: int, sigma: float) -> torch.Tensor:
    """Create 1D Gaussian kernel."""
    x = torch.linspace(-sigma * 3, sigma * 3, kernel_size)
    kernel = torch.exp(-x**2 / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel


# =====================
# Public API Functions
# =====================

def process_zoom_frame(frame: np.ndarray, zoom_factor: float = 1.2) -> np.ndarray:
    """
    Zoom a frame, using GPU if available.
    
    Args:
        frame: Input video frame
        zoom_factor: Zoom factor
        
    Returns:
        Processed frame
    """
    from ..effects.basic_effects import process_zoom_frame as cpu_zoom
    
    return _apply_effect(
        frame, 
        zoom_frame_gpu, 
        use_gpu=gpu_manager.should_use_gpu(), 
        zoom_factor=zoom_factor,
        cpu_fn=cpu_zoom
    )

def process_crop_frame(frame: np.ndarray, crop_percent: float = 0.1) -> np.ndarray:
    """
    Crop a frame, using GPU if available.
    
    Args:
        frame: Input video frame
        crop_percent: Amount to crop from edges
        
    Returns:
        Processed frame
    """
    from ..effects.basic_effects import process_crop_frame as cpu_crop
    
    return _apply_effect(
        frame, 
        crop_frame_gpu, 
        use_gpu=gpu_manager.should_use_gpu(), 
        crop_percent=crop_percent,
        cpu_fn=cpu_crop
    )

def process_filter_frame(frame: np.ndarray, filter_type: str = 'brightness', intensity: float = 0.2) -> np.ndarray:
    """
    Apply filter to a frame, using GPU if available.
    
    Args:
        frame: Input video frame
        filter_type: Type of filter
        intensity: Effect intensity
        
    Returns:
        Processed frame
    """
    from ..effects.basic_effects import process_filter_frame as cpu_filter
    
    return _apply_effect(
        frame, 
        filter_frame_gpu, 
        use_gpu=gpu_manager.should_use_gpu(), 
        filter_type=filter_type,
        intensity=intensity,
        cpu_fn=cpu_filter
    )

def process_blur_frame(frame: np.ndarray, radius: int = 5) -> np.ndarray:
    """
    Apply blur to a frame, using GPU if available.
    
    Args:
        frame: Input video frame
        radius: Blur radius
        
    Returns:
        Processed frame
    """
    # CPU fallback
    def cpu_blur(frame, radius=5, **kwargs):
        import cv2
        return cv2.GaussianBlur(frame, (radius*2+1, radius*2+1), radius/3.0)
    
    return _apply_effect(
        frame, 
        blur_frame_gpu, 
        use_gpu=gpu_manager.should_use_gpu(), 
        radius=radius,
        cpu_fn=cpu_blur
    )

# Function to check if GPU effects are available
def gpu_effects_available() -> bool:
    """Check if GPU effects are available."""
    return TORCH_AVAILABLE and gpu_manager.has_cuda 