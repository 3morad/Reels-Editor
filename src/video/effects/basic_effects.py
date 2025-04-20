from moviepy.editor import VideoFileClip, vfx
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import Optional, Union, Tuple

from ..utils.logging_utils import configure_logger, timed, log_exceptions

# Configure logger
logger = configure_logger("BasicEffects")

# ==========================================
# Per-frame helper functions for parallelism
# ==========================================

def process_zoom_frame(frame: np.ndarray, zoom_factor: float) -> np.ndarray:
    """Zoom a single frame around its center."""
    img = Image.fromarray(frame)
    w, h = img.size
    new_w, new_h = int(w * zoom_factor), int(h * zoom_factor)
    zoomed = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - w) // 2
    top = (new_h - h) // 2
    cropped = zoomed.crop((left, top, left + w, top + h))
    return np.array(cropped)


def process_crop_frame(frame: np.ndarray, crop_percent: float) -> np.ndarray:
    """Crop edges of a single frame by a percentage."""
    img = Image.fromarray(frame)
    w, h = img.size
    dx, dy = int(w * crop_percent), int(h * crop_percent)
    cropped = img.crop((dx, dy, w - dx, h - dy))
    return np.array(cropped)


def process_filter_frame(frame: np.ndarray, filter_type: str, intensity: float) -> np.ndarray:
    """Apply color/blur/brightness filters to a single frame."""
    img = Image.fromarray(frame)
    if filter_type == "grayscale":
        img = img.convert("L").convert("RGB")
    elif filter_type == "blur":
        radius = intensity * 3
        img = img.filter(ImageFilter.GaussianBlur(radius))
    elif filter_type == "brightness":
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(0.5 + intensity * 1.5)
    elif filter_type == "contrast":
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(0.5 + intensity * 1.5)
    elif filter_type == "saturation":
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(intensity * 2)
    elif filter_type == "sepia":
        # Simple sepia via color matrix
        sepia = img.convert("RGB")
        arr = np.array(sepia)
        tr = (arr[:,:,0] * 0.393 + arr[:,:,1] * 0.769 + arr[:,:,2] * 0.189)
        tg = (arr[:,:,0] * 0.349 + arr[:,:,1] * 0.686 + arr[:,:,2] * 0.168)
        tb = (arr[:,:,0] * 0.272 + arr[:,:,1] * 0.534 + arr[:,:,2] * 0.131)
        sepia_arr = np.stack([tr, tg, tb], axis=-1).clip(0,255).astype(np.uint8)
        return sepia_arr
    elif filter_type == "invert":
        img = Image.fromarray(255 - np.array(img))
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")
    return np.array(img)

# ========================
# Original clip-level API
# ========================

@timed
@log_exceptions
def apply_zoom(clip: VideoFileClip, zoom_factor: float = 1.2) -> VideoFileClip:
    """Apply zoom effect to the video clip"""
    if zoom_factor <= 1.0:
        raise ValueError("Zoom factor must be greater than 1.0")
    logger.info(f"Applying zoom effect with factor {zoom_factor}")
    return clip.fx(vfx.resize, zoom_factor)

@timed
@log_exceptions
def apply_crop(clip: VideoFileClip, crop_percent: float = 0.1) -> VideoFileClip:
    """Apply crop effect to the video clip"""
    if not 0 <= crop_percent < 0.5:
        raise ValueError("Crop percent must be between 0 and 0.5")
    logger.info(f"Applying crop effect with {crop_percent*100}% crop")
    return clip.crop(x1=clip.w*crop_percent, y1=clip.h*crop_percent,
                     x2=clip.w*(1-crop_percent), y2=clip.h*(1-crop_percent))

@timed
@log_exceptions
def apply_filter(clip: VideoFileClip, filter_type: str, intensity: float = 1.0) -> VideoFileClip:
    """Apply filter effect to the video clip"""
    if intensity < 0 or intensity > 1:
        raise ValueError("Intensity must be between 0 and 1")
    logger.info(f"Applying {filter_type} filter with intensity {intensity}")
    # Delegate to frame helpers on entire clip
    return clip.fl_image(lambda frame: process_filter_frame(frame, filter_type, intensity))

@timed
@log_exceptions
def apply_transition(clip: VideoFileClip, transition_type: str, duration: float = 1.0) -> VideoFileClip:
    """Apply transition effect to the video clip"""
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
    """Apply trim effect to the video clip"""
    if not 0.1 <= trim_percent <= 0.9:
        raise ValueError("Trim percent must be between 0.1 and 0.9")
    logger.info(f"Applying trim effect with {trim_percent*100}% trim from end")
    new_end = clip.duration * (1 - trim_percent)
    return clip.subclip(0, new_end)