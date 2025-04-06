import numpy as np
from typing import Tuple, Optional, Dict, Any
from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip, concatenate_videoclips
from moviepy.video.fx import all as vfx
import cv2
from .text_processor import TextProcessor
from PIL import Image
import random
import time

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
        print("\n=== VideoTransformer Initialization ===")
        print(f"Input video_clip type: {type(video_clip)}")
        print(f"Input video_clip attributes: {dir(video_clip)}")
        
        if video_clip is None:
            print("ERROR: Video clip is None")
            raise ValueError("Video clip cannot be None")
            
        # Ensure video clip is properly initialized
        if not hasattr(video_clip, 'h') or not hasattr(video_clip, 'w'):
            print("ERROR: Video clip missing height or width attributes")
            print(f"Available attributes: {dir(video_clip)}")
            raise ValueError("Video clip must be properly initialized with height and width")
            
        print(f"Video dimensions: {video_clip.h}x{video_clip.w}")
        print(f"Video duration: {video_clip.duration}")
        print(f"Video fps: {video_clip.fps}")
        
        self.video_clip = video_clip
        self.transformed_clip = None
        self.effects = []
        self.text_processor = TextProcessor()
        print("=== VideoTransformer Initialization Complete ===\n")

    def _validate_frame(self, frame) -> bool:
        """Validate if a frame is a valid numpy array with correct dimensions."""
        print(f"\n=== Frame Validation ===")
        print(f"Frame type: {type(frame)}")
        print(f"Frame attributes: {dir(frame)}")
        
        if not isinstance(frame, np.ndarray):
            print("ERROR: Frame is not a numpy array")
            return False
            
        if len(frame.shape) < 2:
            print(f"ERROR: Frame has insufficient dimensions: {frame.shape}")
            return False
            
        if frame.shape[0] <= 0 or frame.shape[1] <= 0:
            print(f"ERROR: Frame has invalid dimensions: {frame.shape}")
            return False
            
        print(f"Frame shape: {frame.shape}")
        print("=== Frame Validation Complete ===\n")
        return True

    def _process_frame_safely(self, frame, process_func):
        """Safely process a frame with error handling."""
        print("\n=== Safe Frame Processing ===")
        if not self._validate_frame(frame):
            print("ERROR: Frame validation failed")
            return frame
        try:
            result = process_func(frame)
            print("Frame processing successful")
            return result
        except Exception as e:
            print(f"ERROR processing frame: {e}")
            return frame
        print("=== Safe Frame Processing Complete ===\n")

    def apply_zoom(self, zoom_factor: float = 1.2) -> 'VideoTransformer':
        """Apply a zoom effect to the video."""
        print(f"\n=== Applying Zoom Effect (factor: {zoom_factor}) ===")
        if zoom_factor <= 0:
            print("ERROR: Invalid zoom factor")
            raise ValueError("Zoom factor must be positive")
            
        def zoom_frame(frame):
            print(f"\nProcessing zoom frame")
            if not isinstance(frame, np.ndarray):
                print("ERROR: Frame is not a numpy array")
                return frame
                
            try:
                h, w = frame.shape[:2]
                print(f"Original dimensions: {h}x{w}")
                center_x, center_y = w // 2, h // 2
                new_w = int(w / zoom_factor)
                new_h = int(h / zoom_factor)
                x1 = center_x - new_w // 2
                y1 = center_y - new_h // 2
                x2 = x1 + new_w
                y2 = y1 + new_h
                print(f"Zoom region: ({x1}, {y1}) to ({x2}, {y2})")
                zoomed = frame[y1:y2, x1:x2]
                result = cv2.resize(zoomed, (w, h))
                print("Zoom effect applied successfully")
                return result
            except Exception as e:
                print(f"ERROR in zoom effect: {e}")
                return frame
        
        try:
            # Use fl_image without time parameter
            self.transformed_clip = self.video_clip.fl_image(zoom_frame)
            print("Zoom effect applied to video clip")
            self.effects.append({
                'type': 'zoom',
                'zoom_factor': zoom_factor
            })
        except Exception as e:
            print(f"ERROR applying zoom effect: {e}")
        print("=== Zoom Effect Complete ===\n")
        return self

    def apply_crop(self, crop_percent: float = 0.1) -> 'VideoTransformer':
        """Apply a crop effect to the video."""
        print(f"\n=== Applying Crop Effect (percent: {crop_percent}) ===")
        if crop_percent <= 0 or crop_percent >= 0.5:
            print("ERROR: Invalid crop percentage")
            raise ValueError("Crop percent must be between 0 and 0.5")
            
        def crop_frame(frame):
            print(f"\nProcessing crop frame")
            if not self._validate_frame(frame):
                return frame
            try:
                h, w = frame.shape[:2]
                print(f"Original dimensions: {h}x{w}")
                crop_px = int(min(w, h) * crop_percent)
                print(f"Crop pixels: {crop_px}")
                result = frame[crop_px:-crop_px, crop_px:-crop_px]
                print(f"Result dimensions: {result.shape}")
                return result
            except Exception as e:
                print(f"ERROR in crop effect: {e}")
                return frame
        
        try:
            self.transformed_clip = self.video_clip.fl_image(crop_frame)
            print("Crop effect applied to video clip")
            self.effects.append({
                'type': 'crop',
                'crop_percent': crop_percent
            })
        except Exception as e:
            print(f"ERROR applying crop effect: {e}")
        print("=== Crop Effect Complete ===\n")
        return self

    def apply_filter(self, filter_type: str, intensity: float = 1.0) -> 'VideoTransformer':
        """Apply a visual filter to the video."""
        print(f"\n=== Applying Filter Effect (type: {filter_type}, intensity: {intensity}) ===")
        if intensity <= 0:
            print("ERROR: Invalid intensity value")
            raise ValueError("Intensity must be positive")
            
        if filter_type not in ['grayscale', 'blur', 'colorx', 'sepia', 'invert', 'brightness', 'contrast', 'saturation']:
            print(f"ERROR: Unsupported filter type: {filter_type}")
            raise ValueError(f"Unsupported filter type: {filter_type}")
            
        try:
            if filter_type == 'grayscale':
                print("Applying grayscale filter")
                self.transformed_clip = self.video_clip.fx(vfx.blackwhite)
            elif filter_type == 'blur':
                print("Applying blur filter")
                kernel_size = int(intensity * 3)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                print(f"Blur kernel size: {kernel_size}")
                def blur_frame(frame):
                    print(f"\nProcessing blur frame")
                    if not self._validate_frame(frame):
                        return frame
                    try:
                        result = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
                        print("Blur effect applied successfully")
                        return result
                    except Exception as e:
                        print(f"ERROR in blur effect: {e}")
                        return frame
                self.transformed_clip = self.video_clip.fl_image(blur_frame)
            elif filter_type == 'colorx':
                print("Applying colorx filter")
                self.transformed_clip = self.video_clip.fx(vfx.colorx, intensity)
            elif filter_type == 'sepia':
                print("Applying sepia filter")
                self.transformed_clip = self.video_clip.fx(vfx.colorx, 1.1).fx(vfx.blackwhite, [0.3, 0.59, 0.11])
            elif filter_type == 'invert':
                print("Applying invert filter")
                self.transformed_clip = self.video_clip.fx(vfx.invert_colors)
            elif filter_type in ['brightness', 'contrast', 'saturation']:
                print(f"Applying {filter_type} filter")
                self.transformed_clip = self.video_clip.fx(vfx.colorx, intensity)
                
            print("Filter effect applied successfully")
            self.effects.append({
                'type': 'filter',
                'filter_type': filter_type,
                'intensity': intensity
            })
        except Exception as e:
            print(f"ERROR applying filter: {e}")
        print("=== Filter Effect Complete ===\n")
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
            print(f"Error applying transition: {e}")
            return self

        self.effects.append({
            'type': 'transition',
            'transition_type': transition_type,
            'duration': duration
        })
        return self

    def modify_hash(self, method: str = 'pixel') -> 'VideoTransformer':
        """Apply sophisticated modifications to change video hash while preserving visual quality."""
        print(f"\n=== Applying Hash Modification (method: {method}) ===")
        if method not in ['pixel', 'delay', 'watermark', 'dct', 'temporal', 'noise', 'geometric', 'color']:
            print(f"ERROR: Unsupported hash modification method: {method}")
            raise ValueError(f"Unsupported hash modification method: {method}")
            
        try:
            print(f"Video clip type: {type(self.video_clip)}")
            print(f"Video clip attributes: {dir(self.video_clip)}")
            
            # Use the current transformed clip if it exists, otherwise use the original
            current_clip = self.transformed_clip if self.transformed_clip is not None else self.video_clip
            
            if method == 'dct':
                def modify_frame(frame):
                    if not isinstance(frame, np.ndarray):
                        return frame
                        
                    try:
                        # Ensure dimensions are even for DCT
                        h, w = frame.shape[:2]
                        if h % 2 != 0:
                            h -= 1
                        if w % 2 != 0:
                            w -= 1
                        frame = frame[:h, :w]
                        
                        # Convert to float32 for DCT
                        float_frame = frame.astype(np.float32)
                        modified = np.zeros_like(float_frame)
                        
                        # Process each color channel
                        for i in range(3):
                            # Apply DCT
                            dct = cv2.dct(float_frame[:,:,i])
                            
                            # Modify high-frequency coefficients more aggressively
                            # Focus on coefficients that affect perceptual hashing
                            mask = np.ones(dct.shape)
                            # Increase the modification range and area
                            mask[3:, 3:] = 1 + np.random.uniform(-0.05, 0.05, mask[3:, 3:].shape)
                            dct = dct * mask
                            
                            # Apply inverse DCT
                            modified[:,:,i] = cv2.idct(dct)
                            
                        # Ensure values are in valid range
                        modified = np.clip(modified, 0, 255).astype(np.uint8)
                        
                        # Resize back to original dimensions if needed
                        if modified.shape[:2] != frame.shape[:2]:
                            modified = cv2.resize(modified, (frame.shape[1], frame.shape[0]))
                            
                        return modified
                    except Exception as e:
                        print(f"ERROR in DCT modification: {e}")
                        return frame
                
                self.transformed_clip = current_clip.fl_image(modify_frame)
                
            elif method == 'temporal':
                try:
                    # Create a function to modify frames
                    def modify_frame(frame):
                        try:
                            # Apply multiple subtle modifications
                            modified = frame.copy()
                            # Apply blur
                            modified = cv2.GaussianBlur(modified, (3, 3), 0.5)
                            # Add slight color shift
                            modified = cv2.cvtColor(modified, cv2.COLOR_BGR2HSV)
                            modified[:,:,0] = (modified[:,:,0] + 1) % 180
                            modified = cv2.cvtColor(modified, cv2.COLOR_HSV2BGR)
                            return modified
                        except Exception as e:
                            print(f"ERROR in temporal frame modification: {e}")
                            return frame
                    
                    # Apply modifications to every 12th frame
                    def process_frame(frame, t):
                        try:
                            frame_index = int(t * current_clip.fps)
                            if frame_index % 12 == 0 and frame_index > 0:
                                return modify_frame(frame)
                            return frame
                        except Exception as e:
                            print(f"ERROR in temporal processing: {e}")
                            return frame
                    
                    # Apply modifications using fl_image
                    self.transformed_clip = current_clip.fl_image(lambda gf, t: process_frame(gf(t), t))
                    
                    # Preserve video attributes
                    if hasattr(current_clip, 'duration'):
                        self.transformed_clip.duration = current_clip.duration
                    if hasattr(current_clip, 'fps'):
                        self.transformed_clip.fps = current_clip.fps
                        
                except Exception as e:
                    print(f"ERROR in temporal modification: {e}")
                    self.transformed_clip = current_clip
                
            elif method == 'noise':
                def modify_frame(frame):
                    if not isinstance(frame, np.ndarray):
                        return frame
                        
                    try:
                        # Convert to float32 for processing
                        frame_float = frame.astype(np.float32)
                        
                        # Generate PRNG-seeded noise with increased variance
                        np.random.seed(int(time.time() * 1000) % 1000000)
                        noise = np.random.normal(0, 1.0, frame.shape).astype(np.float32)
                        
                        # Apply noise more aggressively to less noticeable regions
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        edges = cv2.Canny(gray, 100, 200)
                        noise_mask = 1.0 - (edges / 255.0)
                        noise_mask = cv2.GaussianBlur(noise_mask, (5, 5), 0)
                        
                        # Apply noise with mask
                        for c in range(3):
                            frame_float[:,:,c] += noise[:,:,c] * noise_mask * 2.0
                        
                        # Ensure values are in valid range
                        modified = np.clip(frame_float, 0, 255).astype(np.uint8)
                        return modified
                    except Exception as e:
                        print(f"ERROR in noise modification: {e}")
                        return frame
                
                self.transformed_clip = current_clip.fl_image(modify_frame)
                
            elif method == 'geometric':
                def modify_frame(frame):
                    if not isinstance(frame, np.ndarray):
                        return frame
                        
                    try:
                        h, w = frame.shape[:2]
                        # Apply more aggressive perspective transform
                        src_points = np.float32([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
                        # Increase modification to 2%
                        offset = w * 0.02
                        dst_points = np.float32([
                            [0+offset, 0+offset], 
                            [w-1-offset, 0-offset],
                            [0-offset, h-1+offset],
                            [w-1+offset, h-1-offset]
                        ])
                        
                        # Calculate perspective transform matrix
                        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
                        # Apply transform
                        modified = cv2.warpPerspective(frame, matrix, (w, h))
                        return modified
                    except Exception as e:
                        print(f"ERROR in geometric modification: {e}")
                        return frame
                
                self.transformed_clip = current_clip.fl_image(modify_frame)
                
            elif method == 'color':
                def modify_frame(frame):
                    if not isinstance(frame, np.ndarray):
                        return frame
                        
                    try:
                        # Convert to LAB color space
                        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                        
                        # Modify a and b channels more aggressively
                        lab = lab.astype(np.float32)
                        lab[:,:,1] *= 1.02  # Increased modification to A channel
                        lab[:,:,2] *= 0.98  # Increased modification to B channel
                        
                        # Ensure values are in valid range
                        lab = np.clip(lab, 0, 255).astype(np.uint8)
                        
                        # Convert back to BGR
                        modified = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                        return modified
                    except Exception as e:
                        print(f"ERROR in color modification: {e}")
                        return frame
                
                self.transformed_clip = current_clip.fl_image(modify_frame)
                
            elif method == 'pixel':
                def modify_frame(frame):
                    if not isinstance(frame, np.ndarray):
                        return frame
                        
                    try:
                        # Convert to float32 for processing
                        frame_float = frame.astype(np.float32)
                        
                        # Add more aggressive brightness variation
                        brightness_change = np.random.uniform(-3, 3)
                        frame_float += brightness_change
                        
                        # Add more aggressive noise
                        noise = np.random.normal(0, 1.5, frame.shape).astype(np.float32)
                        frame_float += noise
                        
                        # Ensure values are in valid range
                        frame_float = np.clip(frame_float, 0, 255)
                        
                        # Convert back to uint8
                        modified = frame_float.astype(np.uint8)
                        
                        # Apply more aggressive blur
                        modified = cv2.GaussianBlur(modified, (5, 5), 0.8)
                        
                        return modified
                    except Exception as e:
                        print(f"ERROR in pixel modification: {e}")
                        return frame
                
                self.transformed_clip = current_clip.fl_image(modify_frame)
                
            elif method == 'delay':
                print("Video dimensions: {}x{}".format(current_clip.w, current_clip.h))
                # Create multiple noise frames
                blank_frames = []
                for _ in range(3):  # Add 3 frames instead of 2
                    # Create a more complex noise pattern
                    noise_frame = np.random.randint(0, 4, (current_clip.h, current_clip.w, 3), dtype=np.uint8)
                    # Add some structure to the noise
                    noise_frame[::4, ::4] = np.random.randint(0, 8, noise_frame[::4, ::4].shape, dtype=np.uint8)
                    blank_clip = ImageClip(noise_frame).set_duration(1.0/current_clip.fps)
                    blank_frames.append(blank_clip)
                print("Created noise frames")
                
                # Insert the frames at the beginning
                clips = blank_frames + [current_clip]
                self.transformed_clip = concatenate_videoclips(clips)
                
            elif method == 'watermark':
                # Create a more complex watermark
                watermark_size = (40, 40)  # Larger watermark
                # Create a more complex pattern
                pattern = np.zeros((watermark_size[0], watermark_size[1], 3), dtype=np.uint8)
                # Add multiple patterns
                pattern[::2, ::2] = 255  # Checkerboard
                pattern[1::2, 1::2] = 128  # Secondary pattern
                # Add some random elements
                pattern[np.random.random(watermark_size) > 0.8] = np.random.randint(0, 255, 3)
                watermark_clip = ImageClip(pattern).set_duration(current_clip.duration)
                
                # Place watermark in a random position
                x_pos = np.random.randint(0, max(1, current_clip.w - watermark_size[0]))
                y_pos = np.random.randint(0, max(1, current_clip.h - watermark_size[1]))
                watermark_clip = watermark_clip.set_position((x_pos, y_pos)).set_opacity(0.1)  # Slightly more visible
                print(f"Created watermark at position ({x_pos}, {y_pos})")
                
                # Composite the watermark onto the video
                self.transformed_clip = CompositeVideoClip([current_clip, watermark_clip])
                
            # Preserve video attributes
            if self.transformed_clip is not None:
                if hasattr(current_clip, 'duration'):
                    self.transformed_clip.duration = current_clip.duration
                if hasattr(current_clip, 'fps'):
                    self.transformed_clip.fps = current_clip.fps
                
            self.effects.append({
                'type': 'hash_modification',
                'method': method
            })
            
        except Exception as e:
            print(f"ERROR applying hash modification: {e}")
            
        print("=== Hash Modification Complete ===\n")
        return self

    def relocate_text(self) -> 'VideoTransformer':
        """Relocate text in the video."""
        print("\n=== Starting Text Relocation ===")
        
        def process_text_relocation(frame):
            print(f"\nProcessing text relocation frame")
            if not isinstance(frame, np.ndarray):
                print("ERROR: Received non-numpy array frame")
                return frame
                
            try:
                # Process the frame and get text regions
                processed_frame, text_regions = self.text_processor.process_frame(frame)
                
                if not text_regions:
                    print("No text regions found in frame")
                    return frame
                    
                # Get safe positions for text relocation
                safe_positions = self.text_processor.get_safe_positions(frame, text_regions)
                
                if not safe_positions:
                    print("No safe positions found for text relocation")
                    return frame
                    
                # Create mask and inpaint text
                mask = self.text_processor.create_mask(frame, text_regions)
                inpainted = self.text_processor.inpaint_text(frame, mask)
                
                print("Text relocation successful")
                return inpainted
            except Exception as e:
                print(f"ERROR in text relocation: {e}")
                return frame
        
        try:
            # Use fl_image to process frames
            self.transformed_clip = self.video_clip.fl_image(process_text_relocation)
            
            # Preserve video attributes
            if hasattr(self.video_clip, 'duration'):
                self.transformed_clip.duration = self.video_clip.duration
            if hasattr(self.video_clip, 'fps'):
                self.transformed_clip.fps = self.video_clip.fps
                
            print("Text relocation applied to video clip")
            self.effects.append({
                'type': 'text_relocation'
            })
        except Exception as e:
            print(f"ERROR applying text relocation: {e}")
            
        print("=== Text Relocation Complete ===\n")
        return self

    def get_transformed_clip(self) -> VideoFileClip:
        """Get the transformed video clip."""
        print("\n=== Getting Transformed Clip ===")
        try:
            # If no transformations were applied, return the original clip
            if self.transformed_clip is None:
                print("No transformations applied, returning original clip")
                return self.video_clip
                
            # Ensure duration and fps are preserved
            if hasattr(self.video_clip, 'duration'):
                self.transformed_clip.duration = self.video_clip.duration
            if hasattr(self.video_clip, 'fps'):
                self.transformed_clip.fps = self.video_clip.fps
                
            print("Returning transformed clip")
            return self.transformed_clip
        except Exception as e:
            print(f"ERROR getting transformed clip: {e}")
            return self.video_clip
        finally:
            print("=== Getting Transformed Clip Complete ===\n")

    def get_effects(self) -> list:
        """Get the list of applied effects."""
        print("\n=== Getting Applied Effects ===")
        print(f"Number of effects: {len(self.effects)}")
        for effect in self.effects:
            print(f"Effect: {effect}")
        print("=== Getting Applied Effects Complete ===\n")
        return self.effects

    def reset(self) -> 'VideoTransformer':
        """Reset all transformations."""
        print("\n=== Resetting Transformations ===")
        self.transformed_clip = None
        self.effects = []
        print("Transformations reset successfully")
        print("=== Reset Complete ===\n")
        return self 
        return self 