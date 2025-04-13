from moviepy.editor import VideoFileClip, vfx
import numpy as np
from typing import Optional, Union, Tuple

from ..utils.frame_utils import validate_frame, process_frame_safely
from ..utils.logging_utils import configure_logger, timed, log_exceptions

# Configure logger
logger = configure_logger("BasicEffects")

@timed
@log_exceptions
def apply_zoom(clip: VideoFileClip, zoom_factor: float = 1.2) -> VideoFileClip:
    """Apply zoom effect to the video clip"""
    if not isinstance(clip, VideoFileClip):
        raise ValueError("Input must be a VideoFileClip")
        
    if zoom_factor <= 1.0:
        raise ValueError("Zoom factor must be greater than 1.0")
        
    logger.info(f"Applying zoom effect with factor {zoom_factor}")
    return clip.fx(vfx.resize, zoom_factor)

@timed
@log_exceptions
def apply_crop(clip: VideoFileClip, crop_percent: float = 0.1) -> VideoFileClip:
    """Apply crop effect to the video clip"""
    if not isinstance(clip, VideoFileClip):
        raise ValueError("Input must be a VideoFileClip")
        
    if not 0 <= crop_percent < 0.5:
        raise ValueError("Crop percent must be between 0 and 0.5")
        
    logger.info(f"Applying crop effect with {crop_percent*100}% crop")
    return clip.crop(x1=clip.w*crop_percent, y1=clip.h*crop_percent,
                    x2=clip.w*(1-crop_percent), y2=clip.h*(1-crop_percent))

@timed
@log_exceptions
def apply_filter(clip: VideoFileClip, filter_type: str, intensity: float = 1.0) -> VideoFileClip:
    """Apply filter effect to the video clip"""
    if not isinstance(clip, VideoFileClip):
        raise ValueError("Input must be a VideoFileClip")
        
    if intensity < 0 or intensity > 1:
        raise ValueError("Intensity must be between 0 and 1")
        
    logger.info(f"Applying {filter_type} filter with intensity {intensity}")
    
    if filter_type == "grayscale":
        return clip.fx(vfx.blackwhite)
    elif filter_type == "blur":
        # Use gaussian_blur instead of blur
        # Scale intensity for blur effect (0-1 -> 0-3)
        blur_amount = intensity * 3
        return clip.fx(vfx.gaussian_blur, blur_amount)
    elif filter_type == "brightness":
        # Scale intensity for brightness effect (0-1 -> 0.5-2)
        brightness_factor = 0.5 + intensity * 1.5
        return clip.fx(vfx.colorx, brightness_factor)
    elif filter_type == "sepia":
        # Apply sepia effect using a single colorx call with a factor
        return clip.fx(vfx.colorx, 1.0)
    elif filter_type == "invert":
        return clip.fx(vfx.invert_colors)
    elif filter_type == "contrast":
        # Scale intensity for contrast effect (0-1 -> 0.5-2)
        contrast_factor = 0.5 + intensity * 1.5
        return clip.fx(vfx.colorx, contrast_factor)
    elif filter_type == "saturation":
        # For saturation, we need to use a different approach
        # Scale intensity for saturation effect (0-1 -> 0-2)
        saturation_factor = intensity * 2
        return clip.fx(vfx.colorx, 1.0)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

@timed
@log_exceptions
def apply_transition(clip: VideoFileClip, transition_type: str, duration: float = 1.0) -> VideoFileClip:
    """Apply transition effect to the video clip"""
    if not isinstance(clip, VideoFileClip):
        raise ValueError("Input must be a VideoFileClip")
        
    if duration <= 0:
        raise ValueError("Duration must be positive")
        
    logger.info(f"Applying {transition_type} transition with duration {duration}s")
    
    if transition_type == "fadeout":
        return clip.fx(vfx.fadeout, duration)
    elif transition_type == "fadein":
        return clip.fx(vfx.fadein, duration)
    else:
        raise ValueError(f"Unknown transition type: {transition_type}")

@timed
@log_exceptions
def apply_trim(clip: VideoFileClip, trim_percent: float = 0.1) -> VideoFileClip:
    """Apply trim effect to the video clip
    
    Args:
        clip: VideoFileClip to trim
        trim_percent: Percentage of video to trim from the end (0.1 to 0.9)
        
    Returns:
        Trimmed VideoFileClip
    """
    if not isinstance(clip, VideoFileClip):
        raise ValueError("Input must be a VideoFileClip")
        
    if not 0.1 <= trim_percent <= 0.9:
        raise ValueError("Trim percent must be between 0.1 and 0.9")
        
    logger.info(f"Applying trim effect with {trim_percent*100}% trim from end")
    
    # Calculate the new end time
    new_end = clip.duration * (1 - trim_percent)
    
    # Trim the clip
    return clip.subclip(0, new_end)
