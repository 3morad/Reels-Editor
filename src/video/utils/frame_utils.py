import numpy as np
import cv2
from PIL import Image
from typing import Callable, Any, Optional

from .logging_utils import configure_logger, timed, log_exceptions

# Configure logger
logger = configure_logger("FrameUtils")

# Add Pillow compatibility for different versions
try:
    # Try newer Pillow version constants
    RESAMPLING_FILTER = Image.Resampling.LANCZOS
except AttributeError:
    try:
        # Try older Pillow version constants
        RESAMPLING_FILTER = Image.LANCZOS
    except AttributeError:
        # Fallback to oldest version
        RESAMPLING_FILTER = Image.ANTIALIAS

@timed
@log_exceptions
def validate_frame(frame: np.ndarray) -> bool:
    """Validate if a frame is properly formatted"""
    if not isinstance(frame, np.ndarray):
        logger.error("Frame is not a numpy array")
        return False
        
    if frame.ndim != 3:
        logger.error(f"Frame has incorrect number of dimensions: {frame.ndim}")
        return False
        
    if frame.shape[2] != 3:
        logger.error(f"Frame has incorrect number of channels: {frame.shape[2]}")
        return False
        
    if frame.dtype != np.uint8:
        logger.error(f"Frame has incorrect dtype: {frame.dtype}")
        return False
        
    return True

@timed
@log_exceptions
def process_frame_safely(frame: np.ndarray, processor: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Process a frame safely with error handling, bounds‐checking and dtype normalization."""
    if not validate_frame(frame):
        logger.error("Invalid input frame")
        return frame

    try:
        # 1) run your processor
        result = processor(frame)

        # 2) coerce into a NumPy array (no dtype yet)
        result = np.asarray(result)

        # 3) shape check
        if result.ndim != 3 or result.shape[2] != 3:
            logger.error(f"Processor returned wrong shape: {result.shape}")
            return frame

        # 4) promote to a signed or float type so negatives are representable
        if np.issubdtype(result.dtype, np.integer):
            temp = result.astype(np.int16, copy=False)
        elif np.issubdtype(result.dtype, np.floating):
            temp = result.astype(np.float32, copy=False)
            # if floats in [0.0,1.0], scale up
            if temp.max() <= 1.0:
                temp *= 255.0
        else:
            # any other dtype, just go via float32
            temp = result.astype(np.float32)

        # 5) clip all values into [0,255]
        temp = np.clip(temp, 0, 255)

        # 6) single, safe cast to uint8
        safe_frame = temp.astype(np.uint8, copy=False)

        # final sanity check
        if not validate_frame(safe_frame):
            logger.error("Frame invalid after normalization")
            return frame

        return safe_frame

    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        return frame

@timed
@log_exceptions
def pil_to_numpy(pil_image: Image.Image) -> Optional[np.ndarray]:
    """
    Convert PIL Image to numpy array with proper cleanup and validation.
    
    Args:
        pil_image: PIL Image to convert
        
    Returns:
        numpy array or None if conversion fails
        
    Features:
    - Proper PIL image cleanup
    - Type validation
    - Error handling
    """
    if not isinstance(pil_image, Image.Image):
        logger.error("Input is not a PIL Image")
        return None
        
    try:
        np_array = np.array(pil_image)
        # Convert to uint8 if needed
        if np_array.dtype != np.uint8:
            np_array = (np_array * 255).clip(0, 255).astype(np.uint8)
        return np_array
    except Exception as e:
        logger.error(f"Error converting PIL image to numpy: {str(e)}")
        return None
    finally:
        try:
            pil_image.close()
        except:
            pass

@timed
@log_exceptions
def numpy_to_pil(np_array: np.ndarray) -> Optional[Image.Image]:
    """
    Convert numpy array to PIL Image with validation.
    
    Args:
        np_array: Numpy array to convert
        
    Returns:
        PIL Image or None if conversion fails
        
    Features:
    - Input validation
    - Proper color mode handling
    - Error handling
    """
    if not isinstance(np_array, np.ndarray):
        logger.error("Input is not a numpy array")
        return None
        
    try:
        # Handle different array shapes
        if len(np_array.shape) == 2:
            mode = 'L'  # Grayscale
        elif len(np_array.shape) == 3 and np_array.shape[2] == 3:
            mode = 'RGB'
        elif len(np_array.shape) == 3 and np_array.shape[2] == 4:
            mode = 'RGBA'
        else:
            logger.error(f"Invalid array shape for PIL conversion: {np_array.shape}")
            return None
            
        # Ensure uint8 type
        if np_array.dtype != np.uint8:
            np_array = (np_array * 255).clip(0, 255).astype(np.uint8)
            
        return Image.fromarray(np_array, mode)
    except Exception as e:
        logger.error(f"Error converting numpy to PIL: {str(e)}")
        return None
