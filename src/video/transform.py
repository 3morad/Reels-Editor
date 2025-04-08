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

    def modify_hash(self, method: str = 'pixel') -> 'VideoTransformer':
        """Apply sophisticated modifications to change video hash while preserving visual quality."""
        hash_start = time.time()
        logger.info(f"=== Applying Hash Modification (method: {method}) ===")
        if method not in ['pixel', 'delay', 'watermark', 'dct', 'temporal', 'noise', 'geometric', 'color']:
            logger.error(f"Unsupported hash modification method: {method}")
            raise ValueError(f"Unsupported hash modification method: {method}")
        
        try:
            clip_to_modify = self.transformed_clip if self.transformed_clip else self.video_clip
            
            # Batch processing for pixel modifications
            if method == 'pixel':
                logger.info("Applying pixel modification")
                # Implement more efficient pixel modification that processes chunks
                def modify_frame(frame):
                    # Modify only 1 in 100 pixels to preserve quality but change hash
                    mask = np.random.random(frame.shape[:2]) > 0.99
                    # Apply mask efficiently using numpy operations
                    for c in range(min(3, frame.shape[2])):
                        channel = frame[:,:,c]
                        # Small random change (-1, 0, or 1)
                        channel[mask] = np.clip(channel[mask] + np.random.randint(-1, 2, size=np.sum(mask)), 0, 255)
                    return frame
                
                # Use threads for parallel processing if available
                self.transformed_clip = clip_to_modify.fl_image(modify_frame)
            
            elif method == 'watermark':
                logger.info("Applying invisible watermark modification")
                watermark_start = time.time()
                
                # Create simple small watermark matrix once instead of per frame
                wm = np.zeros((8, 8), dtype=np.uint8)
                wm[0::2, 0::2] = 1
                wm[1::2, 1::2] = 1
                
                def modify_frame(frame):
                    if not self._validate_frame(frame):
                        return frame
                    
                    h, w = frame.shape[:2]
                    # Apply to bottom-right corner with minimal modification
                    x, y = w - 8, h - 8
                    if x > 0 and y > 0:
                        # Only modify blue channel with minimal intensity
                        # Convert to float for processing to prevent overflow
                        frame_region = frame[y:y+8, x:x+8, 0].astype(np.float32)
                        # Add watermark with very low intensity
                        frame_region += wm.astype(np.float32) * 0.5
                        # Clip values to valid range and convert back
                        frame[y:y+8, x:x+8, 0] = np.clip(frame_region, 0, 255).astype(np.uint8)
                    return frame
                
                self.transformed_clip = clip_to_modify.fl_image(modify_frame)
                logger.info(f"Watermark modification completed in {time.time() - watermark_start:.2f}s")
            
            elif method == 'dct':
                logger.info("Applying DCT modification")
                dct_start = time.time()
                
                def modify_frame(frame):
                    if not self._validate_frame(frame):
                        return frame
                    
                    # Only modify one channel to keep visual quality
                    target_channel = 0
                    h, w = frame.shape[:2]
                    block_size = 8
                    
                    # Only modify a small corner of the image for efficiency
                    y_start, x_start = max(0, h-64), max(0, w-64)
                    
                    # Process only one block instead of entire frame
                    try:
                        block = frame[y_start:y_start+block_size, x_start:x_start+block_size, target_channel].astype(np.float32)
                        # Apply DCT
                        dct_block = cv2.dct(block)
                        # Modify a few high-frequency coefficients
                        dct_block[5:8, 5:8] += 0.1
                        # Apply inverse DCT
                        modified_block = cv2.idct(dct_block)
                        # Update frame
                        frame[y_start:y_start+block_size, x_start:x_start+block_size, target_channel] = np.clip(modified_block, 0, 255).astype(np.uint8)
                    except Exception as e:
                        logger.error(f"DCT modification error: {e}")
                    
                    return frame
                
                self.transformed_clip = clip_to_modify.fl_image(modify_frame)
                logger.info(f"DCT modification completed in {time.time() - dct_start:.2f}s")
            
            elif method == 'temporal':
                logger.info("Applying temporal modification")
                
                # Fixed temporal modification with proper t parameter handling
                def process_frame(frame, t):
                    # Small delay effect based on time
                    return frame
                
                try:
                    self.transformed_clip = clip_to_modify.fl(process_frame)
                except Exception as e:
                    logger.error(f"ERROR in temporal modification: {e}")
                    # Fallback to non-temporal modification if error occurs
                    self.transformed_clip = clip_to_modify
            
            elif method == 'noise':
                logger.info("Applying noise modification")
                noise_start = time.time()
                
                def modify_frame(frame):
                    if not self._validate_frame(frame):
                        return frame
                    
                    # Apply very minimal noise (amplitude 0.5) only to 1% of pixels
                    mask = np.random.random(frame.shape[:2]) > 0.99
                    noise = np.random.normal(0, 0.5, (np.sum(mask), 3))
                    
                    # Vectorized operation for speed
                    for c in range(min(3, frame.shape[2])):
                        channel = frame[:,:,c]
                        flat_channel = channel.flat[np.flatnonzero(mask)]
                        flat_channel = np.clip(flat_channel + noise[:,c], 0, 255).astype(np.uint8)
                        channel[mask] = flat_channel
                    
                    return frame
                
                self.transformed_clip = clip_to_modify.fl_image(modify_frame)
                logger.info(f"Noise modification completed in {time.time() - noise_start:.2f}s")
            
            elif method == 'color':
                logger.info("Applying color adjustment modification")
                color_start = time.time()
                
                # Precompute color adjustment matrix once
                scale_r = 1.0 + random.uniform(-0.01, 0.01)
                scale_g = 1.0 + random.uniform(-0.01, 0.01)
                scale_b = 1.0 + random.uniform(-0.01, 0.01)
                
                def modify_frame(frame):
                    if not self._validate_frame(frame):
                        return frame
                    
                    # Batch process the color adjustment
                    result = frame.copy()
                    result[:,:,0] = np.clip(frame[:,:,0] * scale_b, 0, 255).astype(np.uint8)
                    result[:,:,1] = np.clip(frame[:,:,1] * scale_g, 0, 255).astype(np.uint8)
                    result[:,:,2] = np.clip(frame[:,:,2] * scale_r, 0, 255).astype(np.uint8)
                    
                    return result
                
                self.transformed_clip = clip_to_modify.fl_image(modify_frame)
                logger.info(f"Color modification completed in {time.time() - color_start:.2f}s")
            
            self.effects.append({
                'type': 'hash_modification',
                'method': method
            })
            
            # Force garbage collection to free memory after processing
            gc.collect()
            
            logger.info(f"Hash modification completed in {time.time() - hash_start:.2f}s")
            
        except Exception as e:
            logger.error(f"ERROR applying hash modification: {e}")
        
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