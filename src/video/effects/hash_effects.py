from moviepy.editor import VideoFileClip
import cv2
import numpy as np
import random
import time
import logging
import traceback
from scipy.fftpack import dct, idct
from typing import Optional, Union, Tuple
from ..utils.frame_utils import validate_frame, process_frame_safely
from ..utils.logging_utils import configure_logger, timed, log_exceptions

# Configure logger
logger = configure_logger("HashEffects")

@timed
@log_exceptions
def modify_hash(clip: VideoFileClip, hash_type: str, intensity: float = 1.0) -> VideoFileClip:
    """Apply hash effect to the video clip"""
    if not isinstance(clip, VideoFileClip):
        raise ValueError("Input must be a VideoFileClip")
        
    if not 0 <= intensity <= 1.0:
        raise ValueError("Intensity must be between 0 and 1")
        
    logger.info(f"Applying {hash_type} hash effect with intensity {intensity}")
    
    # Map of available hash types to their modification functions
    hash_functions = {
        "pixelate": _apply_pixel_modification,
        "glitch": _apply_glitch_modification,
        "dct": _apply_dct_modification,
        "delay": _apply_delay_modification,
        "watermark": _apply_watermark_modification,
        "temporal": _apply_temporal_modification,
        "noise": _apply_noise_modification,
        "color": _apply_color_modification,
        "metadata": _apply_metadata_modification
    }
    
    if hash_type not in hash_functions:
        raise ValueError(f"Unknown hash type: {hash_type}. Available types: {list(hash_functions.keys())}")
        
    # Apply the selected modification function
    return hash_functions[hash_type](clip, intensity)

@timed
@log_exceptions
def _apply_pixel_modification(clip: VideoFileClip, intensity: float) -> VideoFileClip:
    """Apply pixel-level modification"""
    def pixel_modifier(frame):
        h, w = frame.shape[:2]
        # Create deterministic pixel pattern with higher intensity but fewer pixels
        # Use a fixed pattern instead of random to improve performance
        step = max(1, int(10 / intensity))  # Adjust step size based on intensity
        for y in range(0, h, step):
            for x in range(0, w, step):
                if (x + y) % 3 == 0:  # Deterministic pattern
                    # Modify a small block of pixels
                    y_end = min(y + 2, h)
                    x_end = min(x + 2, w)
                    frame[y:y_end, x:x_end, 0] = np.clip(frame[y:y_end, x:x_end, 0] + 1, 0, 255)
        return frame
    
    return clip.fl_image(lambda f: process_frame_safely(f, pixel_modifier))

@timed
@log_exceptions
def _apply_glitch_modification(clip: VideoFileClip, intensity: float) -> VideoFileClip:
    """Apply glitch effect to the video"""
    def glitch_modifier(frame):
        h, w = frame.shape[:2]
        # Use a fixed glitch amount based on intensity
        glitch_amount = int(w * intensity * 0.05)
        if glitch_amount > 0:
            # Apply glitch only to a portion of the frame for better performance
            frame[:, w//4:3*w//4] = np.roll(frame[:, w//4:3*w//4], glitch_amount, axis=1)
        return frame
    
    return clip.fl_image(lambda f: process_frame_safely(f, glitch_modifier))

@timed
@log_exceptions
def _apply_dct_modification(clip: VideoFileClip, intensity: float) -> VideoFileClip:
    """Apply DCT-based modification"""
    def dct_modifier(frame):
        # Only modify the first channel for better performance
        channel = frame[:,:,0].astype(float)
        coeffs = dct(dct(channel.T, norm='ortho').T, norm='ortho')
        # Modify only a small portion of coefficients
        coeffs[0:4, 0:4] += np.random.normal(0, intensity * 2.0, (4,4))
        frame[:,:,0] = idct(idct(coeffs.T, norm='ortho').T, norm='ortho')
        return np.clip(frame, 0, 255).astype(np.uint8)
    
    return clip.fl_image(lambda f: process_frame_safely(f, dct_modifier))

@timed
@log_exceptions
def _apply_delay_modification(clip: VideoFileClip, intensity: float) -> VideoFileClip:
    """Apply delay modification"""
    # Increase the delay amount
    delay_ms = int(1 + (intensity * 10))  # Increased from 5 to 10
    return clip.set_start(delay_ms/1000.0)

@timed
@log_exceptions
def _apply_watermark_modification(clip: VideoFileClip, intensity: float) -> VideoFileClip:
    """Apply watermark modification"""
    # Use a fixed watermark size for better performance
    wm_size = 24
    
    def watermark_modifier(frame):
        # Create a simple watermark pattern
        wm = np.zeros((wm_size, wm_size))
        wm[::2, ::2] = intensity * 1.0
        
        # Apply watermark to frame
        frame[0:wm_size, 0:wm_size] += wm * 50
        
        return frame
    
    return clip.fl_image(lambda f: process_frame_safely(f, watermark_modifier))

@timed
@log_exceptions
def _apply_temporal_modification(clip: VideoFileClip, intensity: float) -> VideoFileClip:
    """Apply temporal modification"""
    def temporal_modifier(frame, t):
        if t % 1.0 < 0.1:  # Modify every second
            return frame * (1 + intensity)
        return frame
    
    return clip.fl_image(lambda frame, t: process_frame_safely(frame, lambda x: temporal_modifier(x, t)))

@timed
@log_exceptions
def _apply_noise_modification(clip: VideoFileClip, intensity: float) -> VideoFileClip:
    """Apply noise modification"""
    def noise_modifier(frame):
        # Apply noise only to a portion of the frame for better performance
        h, w = frame.shape[:2]
        noise = np.random.normal(0, intensity * 2.0, (h//2, w//2, 3))
        frame[h//4:3*h//4, w//4:3*w//4] = np.clip(frame[h//4:3*h//4, w//4:3*w//4] + noise, 0, 255)
        return frame
    
    return clip.fl_image(lambda f: process_frame_safely(f, noise_modifier))

@timed
@log_exceptions
def _apply_color_modification(clip: VideoFileClip, intensity: float) -> VideoFileClip:
    """Apply color modification"""
    def color_modifier(frame):
        # Create a simple pattern that affects the hash
        h, w = frame.shape[:2]
        pattern = np.zeros((h, w), dtype=np.uint8)
        
        # Create a pattern with fewer iterations
        step = max(5, int(15 / intensity))  # Adjust step size based on intensity
        for i in range(0, h, step):
            for j in range(0, w, step):
                if (i + j) % 3 == 0:  # Deterministic pattern
                    pattern[i:i+3, j:j+3] = 1
        
        # Apply the pattern to a single channel
        frame[:,:,0] = np.clip(frame[:,:,0] + pattern, 0, 255)
        
        return frame
    
    return clip.fl_image(lambda f: process_frame_safely(f, color_modifier))

@timed
@log_exceptions
def _apply_metadata_modification(clip: VideoFileClip, intensity: float) -> VideoFileClip:
    """Apply metadata modifications to help avoid detection."""
    logger.info(f"Applying metadata modifications with intensity {intensity}")
    
    # Subtle FPS modification
    fps_mod = clip.fps + (random.uniform(-0.1, 0.1) * intensity)
    clip = clip.set_fps(fps_mod)
    
    # Subtle resolution modification (maintaining aspect ratio)
    scale = 1 + (random.uniform(-0.01, 0.01) * intensity)
    new_w = int(clip.w * scale)
    new_h = int(clip.h * scale)
    clip = clip.resize(width=new_w, height=new_h)
    
    # Modify video properties
    clip = clip.set_duration(clip.duration + (random.uniform(-0.001, 0.001) * intensity))
    
    # Add random rotation (very subtle)
    rotation = random.uniform(-0.1, 0.1) * intensity
    if abs(rotation) > 0.01:  # Only apply if rotation is significant enough
        clip = clip.rotate(rotation)
    
    return clip
