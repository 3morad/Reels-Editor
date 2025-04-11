import numpy as np
from typing import Tuple, Optional, Dict, Any
from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip, concatenate_videoclips
from moviepy.video.fx import all as vfx
import cv2
from .text_processor import TextProcessor
from PIL import Image
import random
import time
import logging
import gc
import traceback
import os
from datetime import datetime

# Configure logging
logger = logging.getLogger("VideoTransformer")

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

class VideoTransformer:
    def __init__(self, video_clip: VideoFileClip):
        self.start_time = time.time()
        logger.info("=== VideoTransformer Initialization ===")
        logger.debug(f"Input video_clip type: {type(video_clip)}")
        
        if video_clip is None:
            logger.error("ERROR: Video clip is None")
            raise ValueError("Video clip cannot be None")
            
        # Ensure video clip is properly initialized
        if not hasattr(video_clip, 'h') or not hasattr(video_clip, 'w'):
            logger.error("ERROR: Video clip missing height or width attributes")
            raise ValueError("Video clip must be properly initialized with height and width")
            
        logger.info(f"Video dimensions: {video_clip.h}x{video_clip.w}")
        logger.info(f"Video duration: {video_clip.duration}s")
        logger.info(f"Video fps: {video_clip.fps}")
        
        # Use lower resolution for processing if high-res
        if video_clip.h > 1080 or video_clip.w > 1920:
            logger.info("High resolution video detected, using resize for faster processing")
            # Determine scale factor while maintaining aspect ratio
            scale = min(1080 / video_clip.h, 1920 / video_clip.w)
            new_h = int(video_clip.h * scale)
            new_w = int(video_clip.w * scale)
            logger.info(f"Resizing video to {new_w}x{new_h} for processing")
            self.video_clip = video_clip.resize(height=new_h, width=new_w)
            self.original_clip = video_clip  # Keep original for export if needed
        else:
            self.video_clip = video_clip
            self.original_clip = None
            
        self.transformed_clip = None
        self.effects = []
        logger.info(f"Initialization completed in {time.time() - self.start_time:.2f}s")

    def _validate_frame(self, frame) -> bool:
        """Validate if a frame is a valid numpy array with correct dimensions."""
        if not isinstance(frame, np.ndarray):
            logger.error("Frame is not a numpy array")
            return False
            
        if len(frame.shape) < 2:
            logger.error(f"Frame has insufficient dimensions: {frame.shape}")
            return False
            
        if frame.shape[0] <= 0 or frame.shape[1] <= 0:
            logger.error(f"Frame has invalid dimensions: {frame.shape}")
            return False
            
        return True

    def _process_frame_safely(self, frame, process_func):
        """Safely process a frame with error handling."""
        if not self._validate_frame(frame):
            logger.error("Frame validation failed")
            return frame
        try:
            result = process_func(frame)
            return result
        except Exception as e:
            logger.error(f"ERROR processing frame: {e}")
            return frame

    # Optimized method to handle PIL images properly
    def _pil_to_numpy(self, pil_image):
        """Safely convert PIL image to numpy array with proper cleanup."""
        try:
            np_array = np.array(pil_image)
            # Explicitly close PIL image to prevent resource leaks
            pil_image.close()
            return np_array
        except Exception as e:
            logger.error(f"Error converting PIL image to numpy: {e}")
            try:
                pil_image.close()
            except:
                pass
            return None

    # Optimized method to handle numpy to PIL conversion
    def _numpy_to_pil(self, np_array):
        """Safely convert numpy array to PIL image."""
        try:
            return Image.fromarray(np_array.astype('uint8'))
        except Exception as e:
            logger.error(f"Error converting numpy to PIL: {e}")
            return None

    def apply_zoom(self, zoom_factor: float = 1.2) -> 'VideoTransformer':
        """Apply a zoom effect to the video."""
        logger.info(f"\n=== Applying Zoom Effect (factor: {zoom_factor}) ===")
        if zoom_factor <= 0:
            logger.error("ERROR: Invalid zoom factor")
            raise ValueError("Zoom factor must be positive")
            
        def zoom_frame(frame):
            logger.info(f"\nProcessing zoom frame")
            if not isinstance(frame, np.ndarray):
                logger.error("ERROR: Frame is not a numpy array")
                return frame
                
            try:
                h, w = frame.shape[:2]
                logger.info(f"Original dimensions: {h}x{w}")
                center_x, center_y = w // 2, h // 2
                new_w = int(w / zoom_factor)
                new_h = int(h / zoom_factor)
                x1 = center_x - new_w // 2
                y1 = center_y - new_h // 2
                x2 = x1 + new_w
                y2 = y1 + new_h
                logger.info(f"Zoom region: ({x1}, {y1}) to ({x2}, {y2})")
                zoomed = frame[y1:y2, x1:x2]
                result = cv2.resize(zoomed, (w, h))
                logger.info("Zoom effect applied successfully")
                return result
            except Exception as e:
                logger.error(f"ERROR in zoom effect: {e}")
                return frame
        
        try:
            # Use fl_image without time parameter
            self.transformed_clip = self.video_clip.fl_image(zoom_frame)
            logger.info("Zoom effect applied to video clip")
            self.effects.append({
                'type': 'zoom',
                'zoom_factor': zoom_factor
            })
        except Exception as e:
            logger.error(f"ERROR applying zoom effect: {e}")
        logger.info("=== Zoom Effect Complete ===\n")
        return self

    def apply_crop(self, crop_percent: float = 0.1) -> 'VideoTransformer':
        """Apply a crop effect to the video."""
        logger.info(f"\n=== Applying Crop Effect (percent: {crop_percent}) ===")
        if crop_percent <= 0 or crop_percent >= 0.5:
            logger.error("ERROR: Invalid crop percentage")
            raise ValueError("Crop percent must be between 0 and 0.5")
            
        def crop_frame(frame):
            logger.info(f"\nProcessing crop frame")
            if not self._validate_frame(frame):
                return frame
            try:
                h, w = frame.shape[:2]
                logger.info(f"Original dimensions: {h}x{w}")
                crop_px = int(min(w, h) * crop_percent)
                logger.info(f"Crop pixels: {crop_px}")
                result = frame[crop_px:-crop_px, crop_px:-crop_px]
                logger.info(f"Result dimensions: {result.shape}")
                return result
            except Exception as e:
                logger.error(f"ERROR in crop effect: {e}")
                return frame
        
        try:
            self.transformed_clip = self.video_clip.fl_image(crop_frame)
            logger.info("Crop effect applied to video clip")
            self.effects.append({
                'type': 'crop',
                'crop_percent': crop_percent
            })
        except Exception as e:
            logger.error(f"ERROR applying crop effect: {e}")
        logger.info("=== Crop Effect Complete ===\n")
        return self

    def apply_filter(self, filter_type: str, intensity: float = 1.0) -> 'VideoTransformer':
        """Apply a visual filter to the video."""
        logger.info(f"\n=== Applying Filter Effect (type: {filter_type}, intensity: {intensity}) ===")
        if intensity <= 0:
            logger.error("ERROR: Invalid intensity value")
            raise ValueError("Intensity must be positive")
            
        if filter_type not in ['grayscale', 'blur', 'colorx', 'sepia', 'invert', 'brightness', 'contrast', 'saturation']:
            logger.error(f"ERROR: Unsupported filter type: {filter_type}")
            raise ValueError(f"Unsupported filter type: {filter_type}")
            
        try:
            if filter_type == 'grayscale':
                logger.info("Applying grayscale filter")
                self.transformed_clip = self.video_clip.fx(vfx.blackwhite)
            elif filter_type == 'blur':
                logger.info("Applying blur filter")
                kernel_size = int(intensity * 3)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                logger.info(f"Blur kernel size: {kernel_size}")
                def blur_frame(frame):
                    logger.info(f"\nProcessing blur frame")
                    if not self._validate_frame(frame):
                        return frame
                    try:
                        result = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
                        logger.info("Blur effect applied successfully")
                        return result
                    except Exception as e:
                        logger.error(f"ERROR in blur effect: {e}")
                        return frame
                self.transformed_clip = self.video_clip.fl_image(blur_frame)
            elif filter_type == 'colorx':
                logger.info("Applying colorx filter")
                self.transformed_clip = self.video_clip.fx(vfx.colorx, intensity)
            elif filter_type == 'sepia':
                logger.info("Applying sepia filter")
                self.transformed_clip = self.video_clip.fx(vfx.colorx, 1.1).fx(vfx.blackwhite, [0.3, 0.59, 0.11])
            elif filter_type == 'invert':
                logger.info("Applying invert filter")
                self.transformed_clip = self.video_clip.fx(vfx.invert_colors)
            elif filter_type in ['brightness', 'contrast', 'saturation']:
                logger.info(f"Applying {filter_type} filter")
                self.transformed_clip = self.video_clip.fx(vfx.colorx, intensity)
                
            logger.info("Filter effect applied successfully")
            self.effects.append({
                'type': 'filter',
                'filter_type': filter_type,
                'intensity': intensity
            })
        except Exception as e:
            logger.error(f"ERROR applying filter: {e}")
        logger.info("=== Filter Effect Complete ===\n")
        return self

    def apply_transition(self, transition_type: str, duration: float = 1.0) -> 'VideoTransformer':
        """Apply a transition effect to the video."""
        if duration <= 0:
            raise ValueError("Duration must be positive")
            
        if transition_type not in ['fadein', 'fadeout']:
            raise ValueError(f"Unsupported transition type: {transition_type}")
            
        try:
            if transition_type == 'fadein':
                self.transformed_clip = self.video_clip.fx(vfx.fadein, duration)
            elif transition_type == 'fadeout':
                self.transformed_clip = self.video_clip.fx(vfx.fadeout, duration)
        except Exception as e:
            logger.error(f"Error applying transition: {e}")
            return self

        self.effects.append({
            'type': 'transition',
            'transition_type': transition_type,
            'duration': duration
        })
        return self

    def modify_hash(self, method: str = 'pixel', min_difference: float = 0.05, variation_id: int = 0) -> 'VideoTransformer':
        """Apply sophisticated modifications to change video hash while preserving visual quality."""
        hash_start = time.time()
        logger.info(f"=== Applying Hash Modification (method: {method}, min_difference: {min_difference}, variation: {variation_id}) ===")
        if method not in ['pixel', 'delay', 'watermark', 'dct', 'temporal', 'noise', 'geometric', 'color']:
            logger.error(f"Unsupported hash modification method: {method}")
            raise ValueError(f"Unsupported hash modification method: {method}")
        
        try:
            # Always use the last transformed clip or original if none exists
            clip_to_modify = self.transformed_clip if self.transformed_clip else self.video_clip
            
            # Use a fixed seed based on variation_id for deterministic but different results per variation
            fixed_seed = 42 + variation_id
            random.seed(fixed_seed)
            np.random.seed(fixed_seed)
            
            # Get video dimensions for adaptive modifications
            h, w = clip_to_modify.h, clip_to_modify.w
            
            # Adapt modification intensity based on video resolution
            # Lower resolution videos need more intense modifications
            resolution_factor = min(1.0, (h * w) / (1920 * 1080))
            intensity_scale = 1.5 * (1.0 / max(0.5, resolution_factor))
            
            # Scale modification regions based on video dimensions
            wm_size = max(16, min(32, h // 30, w // 30))
            block_size = max(16, min(32, h // 30, w // 30))
            
            # Batch processing for pixel modifications
            if method == 'pixel':
                logger.info("Applying pixel modification")
                
                # Create a unique but deterministic pattern based on the variation_id
                pattern_type = variation_id % 4  # 4 different pattern types
                
                def modify_frame(frame):
                    # Create a copy to avoid in-place modifications
                    modified = frame.copy()
                    
                    # Use fixed seed for reproducibility within this variation
                    np.random.seed(fixed_seed)
                    
                    # Different patterns based on variation_id
                    if pattern_type == 0:
                        # Grid pattern
                        h, w = frame.shape[:2]
                        grid_size = max(8, min(16, h // 60, w // 60))
                        for i in range(0, h, grid_size):
                            for j in range(0, w, grid_size):
                                if (i//grid_size + j//grid_size) % 2 == 0:
                                    if i+1 < h and j+1 < w:
                                        # Apply a constant offset to specific grid cells
                                        offset = int(3 * intensity_scale)
                                        modified[i:i+1, j:j+1, 0] = np.clip(modified[i:i+1, j:j+1, 0] + offset, 0, 255)
                    
                    elif pattern_type == 1:
                        # Modify 3% of pixels with higher changes
                        mask = np.random.random(frame.shape[:2]) > 0.97
                        for c in range(min(3, frame.shape[2])):
                            channel = modified[:,:,c]
                            changes = np.random.randint(-4, 5, size=np.sum(mask))
                            changes = np.clip(changes * intensity_scale, -10, 10).astype(np.int8)
                            changes[changes == 0] = 1  # Ensure no zero changes
                            channel[mask] = np.clip(channel[mask] + changes, 0, 255)
                    
                    elif pattern_type == 2:
                        # Diagonal pattern
                        h, w = frame.shape[:2]
                        for i in range(min(h, w)):
                            if i % 20 < 5:  # Every 20 pixels, modify 5
                                for c in range(3):
                                    # Different offset per channel
                                    offset = int((c+1) * 2 * intensity_scale) 
                                    if i < h and i < w:
                                        modified[i, i, c] = np.clip(modified[i, i, c] + offset, 0, 255)
                    
                    else:  # pattern_type == 3
                        # Corner modifications (most perceptual hashes are sensitive to corners)
                        h, w = frame.shape[:2]
                        corner_size = max(16, min(h//10, w//10))
                        
                        # Top-left corner
                        modified[0:corner_size, 0:corner_size, 0] = np.clip(
                            modified[0:corner_size, 0:corner_size, 0] + int(2 * intensity_scale), 0, 255)
                        
                        # Bottom-right corner
                        modified[-corner_size:, -corner_size:, 2] = np.clip(
                            modified[-corner_size:, -corner_size:, 2] + int(2 * intensity_scale), 0, 255)
                    
                    return modified
                
                self.transformed_clip = clip_to_modify.fl_image(modify_frame)
            
            elif method == 'delay':
                logger.info("Applying delay modification")
                delay_start = time.time()
                
                # Add a delay proportional to the variation_id
                delay_ms = 5 + (variation_id % 20)
                self.transformed_clip = clip_to_modify.set_start(delay_ms/1000.0)
                
                # Add a matching audio delay
                if hasattr(clip_to_modify, 'audio') and clip_to_modify.audio is not None:
                    self.transformed_clip = self.transformed_clip.set_audio(
                        clip_to_modify.audio.set_start(delay_ms/1000.0)
                    )
                
                logger.info(f"Delay modification completed in {time.time() - delay_start:.2f}s")
            
            elif method == 'watermark':
                logger.info("Applying invisible watermark modification")
                watermark_start = time.time()
                
                # Create unique watermark pattern based on variation_id
                pattern = (variation_id % 4) + 1  # 4 different patterns
                
                # Create watermark matrix scaled to video dimensions
                wm = np.zeros((wm_size, wm_size), dtype=np.uint8)
                
                if pattern == 1:
                    # Checkerboard pattern
                    wm[0::2, 0::2] = 1
                    wm[1::2, 1::2] = 1
                elif pattern == 2:
                    # Horizontal stripes
                    wm[0::2, :] = 1
                elif pattern == 3:
                    # Vertical stripes
                    wm[:, 0::2] = 1
                else:  # pattern == 4
                    # Diagonal pattern
                    for i in range(wm_size):
                        for j in range(wm_size):
                            if (i + j) % 3 == 0:
                                wm[i, j] = 1
                
                def modify_frame(frame):
                    if not self._validate_frame(frame):
                        return frame
                    
                    # Create a copy to avoid in-place modifications
                    modified = frame.copy()
                    h, w = frame.shape[:2]
                    
                    # Apply watermark to different areas based on variation_id
                    locations = [
                        (0, 0),  # Top-left
                        (0, w - wm_size),  # Top-right
                        (h - wm_size, 0),  # Bottom-left
                        (h - wm_size, w - wm_size)  # Bottom-right
                    ]
                    
                    # Select location based on variation
                    loc_idx = variation_id % len(locations)
                    y, x = locations[loc_idx]
                    
                    # Apply watermark if position is valid
                    if x >= 0 and y >= 0 and x + wm_size <= w and y + wm_size <= h:
                        # Intensity varies by variation
                        intensity = 1.0 + (variation_id % 5) * 0.2
                        
                        # Apply to all channels with varying intensity
                        for c in range(3):
                            channel_intensity = intensity * (c + 1) / 3
                            frame_region = modified[y:y+wm_size, x:x+wm_size, c].astype(np.float32)
                            frame_region += wm.astype(np.float32) * (channel_intensity * intensity_scale)
                            modified[y:y+wm_size, x:x+wm_size, c] = np.clip(frame_region, 0, 255).astype(np.uint8)
                    
                    return modified
                
                self.transformed_clip = clip_to_modify.fl_image(modify_frame)
                logger.info(f"Watermark modification completed in {time.time() - watermark_start:.2f}s")
            
            elif method == 'dct':
                logger.info("Applying DCT modification")
                dct_start = time.time()
                
                # Different DCT patterns based on variation_id
                dct_pattern = (variation_id % 5)
                
                def modify_frame(frame):
                    if not self._validate_frame(frame):
                        return frame
                    
                    # Create a copy to avoid in-place modifications
                    modified = frame.copy()
                    h, w = frame.shape[:2]
                    
                    # Multiple blocks positioned strategically around the frame
                    # Most perceptual hashes are more sensitive to certain regions
                    regions = [
                        (h//4, w//4),        # Upper left quadrant
                        (h//4, w*3//4-block_size),   # Upper right quadrant
                        (h*3//4-block_size, w//4),   # Lower left quadrant
                        (h*3//4-block_size, w*3//4-block_size),  # Lower right quadrant
                        (h//2-block_size//2, w//2-block_size//2)  # Center
                    ]
                    
                    # Select regions based on variation
                    region_idx = variation_id % len(regions)
                    y_start, x_start = regions[region_idx]
                    
                    # Ensure blocks are within frame boundaries
                    if 0 <= y_start < h-block_size and 0 <= x_start < w-block_size:
                        for c in range(3):  # Process all channels
                            try:
                                block = modified[y_start:y_start+block_size, x_start:x_start+block_size, c].astype(np.float32)
                                dct_block = cv2.dct(block)
                                
                                # Different DCT modifications based on pattern
                                if dct_pattern == 0:
                                    # Modify low frequency components
                                    dct_block[0:4, 0:4] += 1.0 * intensity_scale
                                elif dct_pattern == 1:
                                    # Modify mid frequency diagonal
                                    for i in range(8):
                                        if i < block_size and i+8 < block_size:
                                            dct_block[i, i+8] += 1.0 * intensity_scale
                                elif dct_pattern == 2:
                                    # Modify high frequency components
                                    high_freq = min(block_size-4, 12)
                                    dct_block[high_freq:, high_freq:] += 1.0 * intensity_scale
                                elif dct_pattern == 3:
                                    # Modify specific frequency bands
                                    band_size = min(4, block_size//4)
                                    for i in range(0, block_size, band_size*2):
                                        if i+band_size <= block_size:
                                            dct_block[i:i+band_size, :] += 1.0 * intensity_scale
                                else:  # dct_pattern == 4
                                    # Modify vertical frequency components
                                    dct_block[:, 4:8] += 1.0 * intensity_scale
                                
                                modified_block = cv2.idct(dct_block)
                                modified[y_start:y_start+block_size, x_start:x_start+block_size, c] = np.clip(modified_block, 0, 255).astype(np.uint8)
                            except Exception as e:
                                logger.error(f"DCT modification error: {e}")
                    
                    return modified
                
                self.transformed_clip = clip_to_modify.fl_image(modify_frame)
                logger.info(f"DCT modification completed in {time.time() - dct_start:.2f}s")
            
            elif method == 'temporal':
                logger.info("Applying temporal modification")
                temporal_start = time.time()
                
                # Use deterministic but different pattern based on variation_id
                pattern_freq = 0.1 + (variation_id % 10) * 0.02
                pattern_amp = 0.5 + (variation_id % 5) * 0.2
                
                def modify_frame(frame, t):
                    if not self._validate_frame(frame):
                        return frame
                    
                    # Create a copy to avoid in-place modifications
                    modified = frame.copy()
                    
                    # Generate temporal effect with variation-specific parameters
                    phase_shift = np.sin(t * pattern_freq + variation_id * 0.1) * pattern_amp * intensity_scale
                    
                    # Apply to regions more likely to affect perceptual hash
                    h, w = frame.shape[:2]
                    region_size = max(32, min(64, h//16, w//16))
                    
                    # Different regions based on variation
                    regions = [
                        (0, 0, region_size),  # Top-left
                        (0, w-region_size, region_size),  # Top-right
                        (h-region_size, 0, region_size),  # Bottom-left
                        (h-region_size, w-region_size, region_size),  # Bottom-right
                        (h//2-region_size//2, w//2-region_size//2, region_size)  # Center
                    ]
                    
                    # Apply to 1 or 2 regions based on variation_id
                    num_regions = 1 + (variation_id % 2)
                    for i in range(num_regions):
                        region_idx = (variation_id + i) % len(regions)
                        y_start, x_start, size = regions[region_idx]
                        
                        if 0 <= y_start < h-size and 0 <= x_start < w-size:
                            # Apply different shifts to different channels
                            for c in range(3):
                                channel_shift = phase_shift * (1.0 + c * 0.2)
                                modified[y_start:y_start+size, x_start:x_start+size, c] = np.clip(
                                    modified[y_start:y_start+size, x_start:x_start+size, c] + channel_shift,
                                    0, 255
                                ).astype(np.uint8)
                    
                    return modified
                
                self.transformed_clip = clip_to_modify.fl(lambda gf, t: modify_frame(gf(t), t))
                logger.info(f"Temporal modification completed in {time.time() - temporal_start:.2f}s")
            
            elif method == 'noise':
                logger.info("Applying noise modification")
                noise_start = time.time()
                
                # Different noise patterns based on variation_id
                noise_pattern = variation_id % 5
                
                def modify_frame(frame):
                    if not self._validate_frame(frame):
                        return frame
                    
                    # Create a copy to avoid in-place modifications
                    modified = frame.copy()
                    h, w = frame.shape[:2]
                    
                    # Reset random seed for consistent noise per frame
                    np.random.seed(fixed_seed)
                    
                    if noise_pattern == 0:
                        # Standard noise - 3% of pixels
                        mask = np.random.random(frame.shape[:2]) > 0.97
                        noise = np.random.normal(0, 1.5 * intensity_scale, (np.sum(mask), 3))
                        
                        for c in range(min(3, frame.shape[2])):
                            channel = modified[:,:,c]
                            flat_channel = channel.flat[np.flatnonzero(mask)]
                            flat_channel = np.clip(flat_channel + noise[:,c], 0, 255).astype(np.uint8)
                            channel[mask] = flat_channel
                    
                    elif noise_pattern == 1:
                        # Structured noise along edges
                        for y in range(0, h, 8):
                            for x in range(0, w, 8):
                                if (x + y) % 16 == 0 and x+1 < w and y+1 < h:
                                    noise_val = np.random.normal(0, 3 * intensity_scale, 3)
                                    for c in range(3):
                                        modified[y, x, c] = np.clip(modified[y, x, c] + noise_val[c], 0, 255).astype(np.uint8)
                    
                    elif noise_pattern == 2:
                        # Regional noise - affects specific regions
                        region_size = max(32, min(64, h//16, w//16))
                        y_start = (variation_id * 17) % max(1, h - region_size)
                        x_start = (variation_id * 23) % max(1, w - region_size)
                        
                        if y_start + region_size <= h and x_start + region_size <= w:
                            noise = np.random.normal(0, 1.0 * intensity_scale, (region_size, region_size, 3))
                            region = modified[y_start:y_start+region_size, x_start:x_start+region_size, :]
                            modified[y_start:y_start+region_size, x_start:x_start+region_size, :] = np.clip(
                                region + noise, 0, 255
                            ).astype(np.uint8)
                    
                    elif noise_pattern == 3:
                        # Gradient noise - stronger in one direction
                        gradient = np.linspace(0, 1.5 * intensity_scale, h).reshape(-1, 1)
                        for c in range(3):
                            noise = np.random.normal(0, 1.0, (h, w)) * gradient
                            modified[:,:,c] = np.clip(modified[:,:,c] + noise, 0, 255).astype(np.uint8)
                    
                    else:  # noise_pattern == 4
                        # Corner noise - affects corners more strongly
                        corner_size = max(16, min(h//8, w//8))
                        corners = [
                            (0, 0),  # Top-left
                            (0, w-corner_size),  # Top-right
                            (h-corner_size, 0),  # Bottom-left
                            (h-corner_size, w-corner_size)  # Bottom-right
                        ]
                        
                        corner_idx = variation_id % 4
                        y, x = corners[corner_idx]
                        
                        if 0 <= y < h and 0 <= x < w and y+corner_size <= h and x+corner_size <= w:
                            noise = np.random.normal(0, 2.0 * intensity_scale, (corner_size, corner_size, 3))
                            corner_region = modified[y:y+corner_size, x:x+corner_size, :]
                            modified[y:y+corner_size, x:x+corner_size, :] = np.clip(
                                corner_region + noise, 0, 255
                            ).astype(np.uint8)
                    
                    return modified
                
                self.transformed_clip = clip_to_modify.fl_image(modify_frame)
                logger.info(f"Noise modification completed in {time.time() - noise_start:.2f}s")
            
            elif method == 'color':
                logger.info("Applying color adjustment modification")
                color_start = time.time()
                
                # Different color adjustments based on variation_id
                color_pattern = variation_id % 5
                
                # Precompute color adjustment parameters based on variation
                if color_pattern == 0:
                    # RGB balance shift
                    scale_r = 1.0 + 0.03 * intensity_scale
                    scale_g = 1.0
                    scale_b = 1.0 - 0.02 * intensity_scale
                elif color_pattern == 1:
                    # Green emphasis
                    scale_r = 1.0
                    scale_g = 1.0 + 0.04 * intensity_scale
                    scale_b = 1.0
                elif color_pattern == 2:
                    # Blue emphasis
                    scale_r = 1.0
                    scale_g = 1.0
                    scale_b = 1.0 + 0.04 * intensity_scale
                elif color_pattern == 3:
                    # Warm tone
                    scale_r = 1.0 + 0.03 * intensity_scale
                    scale_g = 1.0 + 0.01 * intensity_scale
                    scale_b = 1.0 - 0.01 * intensity_scale
                else:  # color_pattern == 4
                    # Cool tone
                    scale_r = 1.0 - 0.01 * intensity_scale
                    scale_g = 1.0 + 0.01 * intensity_scale
                    scale_b = 1.0 + 0.03 * intensity_scale
                
                def modify_frame(frame):
                    if not self._validate_frame(frame):
                        return frame
                    
                    # Create a copy to avoid in-place modifications
                    modified = frame.copy()
                    
                    # Apply color adjustments based on selected pattern
                    if color_pattern < 4:
                        # Global color adjustment
                        modified[:,:,0] = np.clip(frame[:,:,0] * scale_b, 0, 255).astype(np.uint8)
                        modified[:,:,1] = np.clip(frame[:,:,1] * scale_g, 0, 255).astype(np.uint8)
                        modified[:,:,2] = np.clip(frame[:,:,2] * scale_r, 0, 255).astype(np.uint8)
                    else:
                        # Graduated color adjustment (varies by y-coordinate)
                        h, w = frame.shape[:2]
                        for y in range(h):
                            # Gradual change from top to bottom
                            factor = y / h
                            r_scale = 1.0 + (scale_r - 1.0) * factor
                            g_scale = 1.0 + (scale_g - 1.0) * factor
                            b_scale = 1.0 + (scale_b - 1.0) * factor
                            
                            modified[y,:,0] = np.clip(frame[y,:,0] * b_scale, 0, 255).astype(np.uint8)
                            modified[y,:,1] = np.clip(frame[y,:,1] * g_scale, 0, 255).astype(np.uint8)
                            modified[y,:,2] = np.clip(frame[y,:,2] * r_scale, 0, 255).astype(np.uint8)
                    
                    return modified
                
                self.transformed_clip = clip_to_modify.fl_image(modify_frame)
                logger.info(f"Color modification completed in {time.time() - color_start:.2f}s")
            
            # Add to effects list with timestamp and variation info
            self.effects.append({
                'type': 'hash_modification',
                'method': method,
                'variation_id': variation_id,
                'timestamp': time.time()
            })
            
            # Force garbage collection to free memory after processing
            gc.collect()
            
            logger.info(f"Hash modification completed in {time.time() - hash_start:.2f}s")
            
        except Exception as e:
            logger.error(f"ERROR applying hash modification: {e}")
            logger.error(traceback.format_exc())
        
        return self

    def relocate_text(self) -> 'VideoTransformer':
        """Relocate text in the video."""
        logger.info("\n=== Starting Text Relocation ===")
        # Temporarily disable text relocation
        logger.info("Text relocation is currently disabled")
        return self
        """
        def process_text_relocation(frame):
            logger.info(f"\nProcessing text relocation frame")
            if not isinstance(frame, np.ndarray):
                logger.error("ERROR: Received non-numpy array frame")
                return frame
                
            try:
                # Process the frame and get text regions
                processed_frame, text_regions = self.text_processor.process_frame(frame)
                
                if not text_regions:
                    logger.info("No text regions found in frame")
                    return frame
                    
                # Get safe positions for text relocation
                safe_positions = self.text_processor.get_safe_positions(frame, text_regions)
                
                if not safe_positions:
                    logger.info("No safe positions found for text relocation")
                    return frame
                    
                # Create mask and inpaint text
                mask = self.text_processor.create_mask(frame, text_regions)
                inpainted = self.text_processor.inpaint_text(frame, mask)
                
                logger.info("Text relocation successful")
                return inpainted
            except Exception as e:
                logger.error(f"ERROR in text relocation: {e}")
                return frame
        
        try:
            # Use fl_image to process frames
            self.transformed_clip = self.video_clip.fl_image(process_text_relocation)
            
            # Preserve video attributes
            if hasattr(self.video_clip, 'duration'):
                self.transformed_clip.duration = self.video_clip.duration
            if hasattr(self.video_clip, 'fps'):
                self.transformed_clip.fps = self.video_clip.fps
                
            logger.info("Text relocation applied to video clip")
            self.effects.append({
                'type': 'text_relocation'
            })
        except Exception as e:
            logger.error(f"ERROR applying text relocation: {e}")
            
        logger.info("=== Text Relocation Complete ===\n")
        return self
        """

    def get_transformed_clip(self) -> VideoFileClip:
        """Return the transformed video clip."""
        if self.transformed_clip is not None:
            return self.transformed_clip
        return self.video_clip

    def get_effects(self) -> list:
        """Return the list of applied effects."""
        logger.info("\n=== Getting Applied Effects ===")
        logger.info(f"Number of effects: {len(self.effects)}")
        for effect in self.effects:
            logger.info(f"Effect: {effect}")
        logger.info("=== Getting Applied Effects Complete ===\n")
        return self.effects

    def reset(self) -> 'VideoTransformer':
        """Reset transformations."""
        logger.info("\n=== Resetting Transformations ===")
        self.transformed_clip = None
        self.effects = []
        logger.info("Transformations reset successfully")
        logger.info("=== Reset Complete ===\n")
        return self 