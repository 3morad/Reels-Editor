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
    """Process a frame safely with error handling"""
    if not validate_frame(frame):
        logger.error("Invalid frame format")
        return frame
        
    try:
        processed = processor(frame)
        if not validate_frame(processed):
            logger.error("Processor returned invalid frame format")
            return frame
        return processed
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
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
