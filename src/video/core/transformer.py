from moviepy.editor import VideoFileClip
import logging
import time
from typing import Optional, List, Dict, Any
import random

from ..effects import basic_effects, hash_effects
from ..utils import frame_utils
from ..utils.logging_utils import configure_logger, timed, log_exceptions

# Configure logger
logger = configure_logger("VideoTransformer")

class VideoTransformer:
    def __init__(self, video_clip: VideoFileClip):
        self.start_time = time.time()
        logger.info("=== VideoTransformer Initialization ===")
        
        if video_clip is None:
            raise ValueError("Video clip cannot be None")
            
        if not hasattr(video_clip, 'h') or not hasattr(video_clip, 'w'):
            raise ValueError("Video clip must be properly initialized with height and width")
            
        logger.info(f"Video dimensions: {video_clip.h}x{video_clip.w}")
        logger.info(f"Video duration: {video_clip.duration}s")
        logger.info(f"Video fps: {video_clip.fps}")
        
        # Use lower resolution for processing if high-res
        if video_clip.h > 1080 or video_clip.w > 1920:
            logger.info("High resolution video detected, using resize for faster processing")
            scale = min(1080 / video_clip.h, 1920 / video_clip.w)
            new_h = int(video_clip.h * scale)
            new_w = int(video_clip.w * scale)
            logger.info(f"Resizing video to {new_w}x{new_h} for processing")
            self.video_clip = video_clip.resize(height=new_h, width=new_w)
            self.original_clip = video_clip
        else:
            self.video_clip = video_clip
            self.original_clip = None
            
        self.transformed_clip = None
        self.effects = []
        logger.info(f"Initialization completed in {time.time() - self.start_time:.2f}s")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up resources")
        if self.transformed_clip and self.transformed_clip != self.video_clip:
            try:
                self.transformed_clip.close()
            except Exception as e:
                logger.error(f"Error closing transformed clip: {e}")
        if self.original_clip:
            try:
                self.original_clip.close()
            except Exception as e:
                logger.error(f"Error closing original clip: {e}")
        self.transformed_clip = None
        self.original_clip = None
        self.video_clip = None
        self.effects = []

    @timed
    @log_exceptions
    def apply_effect(self, effect_name: str, **kwargs):
        """Apply an effect by name with validation"""
        effect_map = {
            'zoom': (basic_effects.apply_zoom, ['zoom_factor']),
            'crop': (basic_effects.apply_crop, ['crop_percent']),
            'filter': (basic_effects.apply_filter, ['filter_type', 'intensity']),
            'transition': (basic_effects.apply_transition, ['transition_type', 'duration']),
            'hash': (hash_effects.modify_hash, ['hash_type', 'intensity'])
        }

        if effect_name not in effect_map:
            raise ValueError(f"Unknown effect: {effect_name}")

        effect_fn, required_args = effect_map[effect_name]
        missing = [arg for arg in required_args if arg not in kwargs]
        if missing:
            raise ValueError(f"Missing required arguments for {effect_name}: {missing}")

        # Apply the effect to current clip
        current_clip = self.transformed_clip or self.video_clip
        self.transformed_clip = effect_fn(current_clip, **kwargs)
        self.effects.append({'type': effect_name, **kwargs})
        return self

    @timed
    @log_exceptions
    def apply_zoom(self, zoom_factor: float = 1.2) -> 'VideoTransformer':
        """Apply zoom effect to the video"""
        self.transformed_clip = basic_effects.apply_zoom(
            self.transformed_clip or self.video_clip, 
            zoom_factor
        )
        self.effects.append({'type': 'zoom', 'zoom_factor': zoom_factor})
        return self

    @timed
    @log_exceptions
    def apply_crop(self, crop_percent: float = 0.1) -> 'VideoTransformer':
        """Apply crop effect to the video"""
        self.transformed_clip = basic_effects.apply_crop(
            self.transformed_clip or self.video_clip, 
            crop_percent
        )
        self.effects.append({'type': 'crop', 'crop_percent': crop_percent})
        return self

    @timed
    @log_exceptions
    def apply_filter(self, filter_type: str, intensity: float = 1.0) -> 'VideoTransformer':
        """Apply visual filter to the video"""
        self.transformed_clip = basic_effects.apply_filter(
            self.transformed_clip or self.video_clip,
            filter_type,
            intensity
        )
        self.effects.append({'type': 'filter', 'filter_type': filter_type, 'intensity': intensity})
        return self

    @timed
    @log_exceptions
    def apply_transition(self, transition_type: str, duration: float = 1.0) -> 'VideoTransformer':
        """Apply transition effect to the video"""
        self.transformed_clip = basic_effects.apply_transition(
            self.transformed_clip or self.video_clip,
            transition_type,
            duration
        )
        self.effects.append({'type': 'transition', 'transition_type': transition_type, 'duration': duration})
        return self

    @timed
    @log_exceptions
    def apply_trim(self, trim_percent: float = 0.1) -> 'VideoTransformer':
        """Apply trim effect to the video
        
        Args:
            trim_percent: Percentage of video to trim from the end (0.1 to 0.9)
        """
        self.transformed_clip = basic_effects.apply_trim(
            self.transformed_clip or self.video_clip,
            trim_percent
        )
        self.effects.append({'type': 'trim', 'trim_percent': trim_percent})
        return self

    @timed
    @log_exceptions
    def modify_hash(self, hash_type: str, intensity: float = 1.0) -> 'VideoTransformer':
        """Apply hash effect to the video
        
        Available hash types:
        - pixelate: Applies pixel-level modifications
        - glitch: Applies glitch effect with horizontal shifts
        - dct: Applies DCT-based modifications
        - delay: Applies temporal delay
        - watermark: Adds a watermark pattern
        - temporal: Applies temporal modifications
        - noise: Adds noise to the video
        - color: Modifies color channels
        """
        self.transformed_clip = hash_effects.modify_hash(
            self.transformed_clip or self.video_clip,
            hash_type,
            intensity
        )
        self.effects.append({'type': 'hash', 'hash_type': hash_type, 'intensity': intensity})
        return self

    @timed
    @log_exceptions
    def modify_metadata(self, intensity: float = 0.1) -> 'VideoTransformer':
        """Modify video metadata to help avoid detection while maintaining quality.
        
        Args:
            intensity: How aggressive the metadata modifications should be (0.0 to 1.0)
        """
        if not 0 <= intensity <= 1.0:
            raise ValueError("Intensity must be between 0 and 1")
            
        logger.info(f"Applying metadata modifications with intensity {intensity}")
        
        current_clip = self.transformed_clip or self.video_clip
        
        # Subtle FPS modification
        fps_mod = current_clip.fps + (random.uniform(-0.1, 0.1) * intensity)
        current_clip = current_clip.set_fps(fps_mod)
        
        # Subtle resolution modification (maintaining aspect ratio)
        scale = 1 + (random.uniform(-0.01, 0.01) * intensity)
        new_w = int(current_clip.w * scale)
        new_h = int(current_clip.h * scale)
        current_clip = current_clip.resize(width=new_w, height=new_h)
        
        # Modify video properties
        current_clip = current_clip.set_duration(current_clip.duration + (random.uniform(-0.001, 0.001) * intensity))
        
        # Add random rotation (very subtle)
        rotation = random.uniform(-0.1, 0.1) * intensity
        if abs(rotation) > 0.01:  # Only apply if rotation is significant enough
            current_clip = current_clip.rotate(rotation)
        
        self.transformed_clip = current_clip
        self.effects.append({'type': 'metadata', 'intensity': intensity})
        return self

    def get_transformed_clip(self) -> VideoFileClip:
        """Get the final transformed clip"""
        return self.transformed_clip or self.video_clip

    def get_effects(self) -> List[Dict[str, Any]]:
        """Get list of applied effects"""
        return self.effects

    def reset(self) -> 'VideoTransformer':
        """Reset all transformations"""
        if self.transformed_clip and self.transformed_clip != self.video_clip:
            try:
                self.transformed_clip.close()
            except Exception as e:
                logger.error(f"Error closing transformed clip during reset: {e}")
        self.transformed_clip = None
        self.effects = []
        return self
