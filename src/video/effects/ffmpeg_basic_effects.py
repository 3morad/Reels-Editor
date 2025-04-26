"""
FFmpeg basic video effects for common video processing tasks.
These effects include color adjustments, filters, and basic transformations.
"""

from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def get_brightness_contrast_params(brightness: float = 0.0, contrast: float = 1.0) -> List[str]:
    """
    Adjusts video brightness and contrast.
    Args:
        brightness: Brightness adjustment (-1.0 to 1.0)
        contrast: Contrast adjustment (0.0 to 2.0)
    """
    return [
        '-vf', f'eq=brightness={brightness}:contrast={contrast}'
    ]

def get_saturation_params(saturation: float = 1.0) -> List[str]:
    """
    Adjusts video color saturation.
    Args:
        saturation: Saturation multiplier (0.0 to 3.0)
    """
    return [
        '-vf', f'eq=saturation={saturation}'
    ]

def get_grayscale_params() -> List[str]:
    """
    Converts video to grayscale.
    """
    return [
        '-vf', 'format=gray'
    ]

def get_sepia_params() -> List[str]:
    """
    Applies sepia tone effect.
    """
    return [
        '-vf', 'colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131'
    ]

def get_invert_params() -> List[str]:
    """
    Inverts video colors.
    """
    return [
        '-vf', 'negate'
    ]

def get_zoom_params(zoom_factor: float = 1.2) -> List[str]:
    """
    Get FFmpeg filter parameters for zoom effect.
    Args:
        zoom_factor: Zoom factor (must be greater than 1.0)
    """
    zoom_factor = max(1.01, min(2.0, zoom_factor))  # Clamp between 1.01 and 2.0
    
    return [
        '-vf', f'scale=iw*{zoom_factor}:ih*{zoom_factor},crop=iw/{zoom_factor}:ih/{zoom_factor}:iw/2-iw/({zoom_factor}*2):ih/2-ih/({zoom_factor}*2)'
    ]

def get_crop_params(crop_percent: float = 0.1) -> List[str]:
    """
    Get FFmpeg filter parameters for crop effect.
    Args:
        crop_percent: Percentage to crop from edges (0.0 to 0.5)
    """
    crop_percent = max(0.01, min(0.3, crop_percent))  # Clamp between 0.01 and 0.3
    
    scale_factor = 1.0 - (crop_percent * 2)
    
    return [
        '-vf', f'crop=iw*{scale_factor}:ih*{scale_factor}:iw*{crop_percent}:ih*{crop_percent}'
    ]

def get_blur_params(strength: float = 2.0) -> List[str]:
    """
    Applies Gaussian blur effect.
    Args:
        strength: Blur strength (0.1 to 10.0)
    """
    return [
        '-vf', f'gblur=sigma={strength}'
    ]

def get_sharpen_params(strength: float = 1.0) -> List[str]:
    """
    Applies sharpening effect.
    Args:
        strength: Sharpening strength (0.1 to 5.0)
    """
    return [
        '-vf', f'unsharp=5:5:{strength}:5:5:{strength}'
    ]

def get_vignette_params(strength: float = 0.3) -> List[str]:
    """
    Adds vignette effect.
    Args:
        strength: Vignette intensity (0.0 to 1.0)
    """
    return [
        '-vf', f'vignette=angle=PI/4:strength={strength}'
    ]

def get_rotate_params(angle: float) -> List[str]:
    """
    Rotates the video.
    Args:
        angle: Rotation angle in degrees
    """
    return [
        '-vf', f'rotate={angle}*PI/180'
    ]

def get_flip_params(horizontal: bool = False, vertical: bool = False) -> List[str]:
    """
    Flips the video horizontally or vertically.
    Args:
        horizontal: Flip horizontally if True
        vertical: Flip vertically if True
    """
    if horizontal and vertical:
        return ['-vf', 'hflip,vflip']
    elif horizontal:
        return ['-vf', 'hflip']
    elif vertical:
        return ['-vf', 'vflip']
    return []

def get_color_balance_params(r: float = 0.0, g: float = 0.0, b: float = 0.0) -> str:
    """Get FFmpeg filter parameters for color balance adjustment."""
    return f'colorbalance=rs={r}:gs={g}:bs={b}'

def get_denoise_params(strength: float = 0.1) -> str:
    """Get FFmpeg filter parameters for video denoising."""
    strength = max(0.1, min(1.0, strength))  # Clamp between 0.1 and 1.0
    return f'nlmeans=s={strength*10}'

def get_stabilize_params(shakiness: int = 5, accuracy: int = 15) -> str:
    """Get FFmpeg filter parameters for video stabilization."""
    return f'vidstabtransform=smoothing=10:input="transforms.trf",vidstabdetect=shakiness={shakiness}:accuracy={accuracy}'

def get_fade_params(duration: float = 1.0, fade_in: bool = True) -> str:
    """Get FFmpeg filter parameters for fade effect."""
    duration = max(0.1, min(5.0, duration))  # Clamp between 0.1 and 5.0 seconds
    if fade_in:
        return f'fade=t=in:st=0:d={duration}'
    else:
        return f'fade=t=out:st={duration}:d={duration}'

def get_trim_params(trim_percent: float = 0.1, duration: float = None) -> List[str]:
    """
    Get FFmpeg parameters for trimming video.
    This will actually cut the video duration, matching the old MoviePy logic.
    Args:
        trim_percent: Percentage to trim from end (0.1 to 0.9)
        duration: Original video duration in seconds
    """
    trim_percent = max(0.1, min(0.9, trim_percent))  # Clamp between 0.1 and 0.9
    params = []
    if duration is not None:
        new_end = duration * (1 - trim_percent)
        params.extend(['-t', str(new_end)])
    return params

def get_speed_params(speed: float = 1.0) -> str:
    """Get FFmpeg filter parameters for speed effect."""
    speed = max(0.5, min(2.0, speed))  # Clamp between 0.5x and 2.0x
    return f'setpts={1/speed}*PTS'

def get_overlay_text_params(text: str, x: int = 10, y: int = 10, 
                          font_size: int = 24, color: str = 'white') -> List[str]:
    """
    Adds text overlay to video.
    Args:
        text: Text to display
        x: X coordinate
        y: Y coordinate
        font_size: Font size in pixels
        color: Text color
    """
    return [
        '-vf', f"drawtext=text='{text}':x={x}:y={y}:fontsize={font_size}:fontcolor={color}"
    ]

def get_filter_params(filter_type: str, intensity: float = 1.0) -> List[str]:
    """
    Get FFmpeg filter parameters for basic video effects.
    Args:
        filter_type: Type of filter to apply
        intensity: Effect intensity (0.0 to 1.0)
    """
    intensity = max(0.1, min(1.0, intensity))  # Clamp intensity between 0.1 and 1.0
    
    # Apply a reduction factor to make all effects more subtle
    subtle_factor = 0.3  # Reduce intensity by 70%
    intensity = intensity * subtle_factor
    
    if filter_type == 'grayscale':
        return ['-vf', 'format=gray']
    elif filter_type == 'blur':
        radius = intensity   # Reduced from 3 to 2
        return ['-vf', f'gblur=sigma={radius}']
    elif filter_type == 'brightness':
        # Reduce range from 0.5-2.0 to 0.8-1.2
        return ['-vf', f'eq=brightness={0.8 + intensity * 0.4}']
    elif filter_type == 'contrast':
        # Reduce range from 0.5-2.0 to 0.9-1.1
        return ['-vf', f'eq=contrast={0.9 + intensity * 0.2}']
    elif filter_type == 'saturation':
        # Reduce range from 0-2.0 to 0.8-1.2
        return ['-vf', f'eq=saturation={0.8 + intensity * 0.4}']
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

def get_transition_params(transition_type: str, duration: float = 1.0, video_duration: float = None) -> List[str]:
    """
    Get FFmpeg filter parameters for transition effects.
    Args:
        transition_type: Type of transition ('fadein' or 'fadeout')
        duration: Duration of the transition in seconds
        video_duration: Total duration of the video (for fadeout)
    """
    duration = max(0.1, min(5.0, duration))  # Clamp between 0.1 and 5.0 seconds
    if transition_type == 'fadein':
        return ['-vf', f'fade=t=in:st=0:d={duration}']
    elif transition_type == 'fadeout':
        if video_duration is not None:
            start = max(0, video_duration - duration)
        else:
            start = duration  # fallback to old behavior
        return ['-vf', f'fade=t=out:st={start}:d={duration}']
    else:
        raise ValueError(f"Unknown transition type: {transition_type}") 