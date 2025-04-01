import numpy as np
from typing import Tuple, Optional, Dict, Any
from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip, concatenate_videoclips
from moviepy.video.fx import all as vfx
import cv2
from .text_processor import TextProcessor

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
        """Apply subtle modifications to change video hash without affecting quality."""
        print(f"\n=== Applying Hash Modification (method: {method}) ===")
        if method not in ['pixel', 'delay', 'watermark']:
            print(f"ERROR: Unsupported hash modification method: {method}")
            raise ValueError(f"Unsupported hash modification method: {method}")
            
        try:
            print(f"Video clip type: {type(self.video_clip)}")
            print(f"Video clip attributes: {dir(self.video_clip)}")
            
            if method == 'pixel':
                def modify_frame(frame):
                    print(f"\nProcessing hash modification frame")
                    print(f"Frame type: {type(frame)}")
                    print(f"Frame attributes: {dir(frame)}")
                    
                    if not isinstance(frame, np.ndarray):
                        print("ERROR: Frame is not a numpy array")
                        return frame
                        
                    try:
                        frame_copy = frame.copy()
                        print(f"Frame shape: {frame_copy.shape}")
                        frame_copy[0, 0] = [
                            max(0, min(255, frame_copy[0, 0, 0] + 1)),  # R
                            max(0, min(255, frame_copy[0, 0, 1] + 1)),  # G
                            max(0, min(255, frame_copy[0, 0, 2] + 1))   # B
                        ]
                        print("Pixel modification successful")
                        return frame_copy
                    except Exception as e:
                        print(f"ERROR in pixel modification: {e}")
                        return frame
                        
                try:
                    self.transformed_clip = self.video_clip.fl_image(modify_frame)
                    print("Hash modification applied to video clip")
                except Exception as e:
                    print(f"ERROR applying frame modification: {e}")
                    return self
                    
            elif method == 'delay':
                try:
                    if not hasattr(self.video_clip, 'h') or not hasattr(self.video_clip, 'w'):
                        print("ERROR: Video clip missing dimensions")
                        return self
                        
                    h, w = self.video_clip.h, self.video_clip.w
                    print(f"Video dimensions: {h}x{w}")
                    
                    if h <= 0 or w <= 0:
                        print("ERROR: Invalid video dimensions")
                        return self
                        
                    blank_frame = np.zeros((h, w, 3), dtype='uint8')
                    print("Created blank frame")
                    blank_clip = ImageClip(blank_frame).set_duration(1.0/self.video_clip.fps)
                    print("Created blank clip")
                    self.transformed_clip = concatenate_videoclips([blank_clip, self.video_clip])
                    print("Delay modification applied successfully")
                except Exception as e:
                    print(f"ERROR in delay modification: {e}")
                    return self
                    
            elif method == 'watermark':
                try:
                    watermark = np.ones((5, 5, 3), dtype='uint8') * 255
                    print("Created watermark")
                    watermark_clip = ImageClip(watermark).set_opacity(0.01)
                    watermark_clip = watermark_clip.set_position((0, 0))
                    self.transformed_clip = CompositeVideoClip([self.video_clip, watermark_clip])
                    print("Watermark modification applied successfully")
                except Exception as e:
                    print(f"ERROR in watermark modification: {e}")
                    return self
                    
        except Exception as e:
            print(f"ERROR modifying hash: {e}")
            return self
        
        self.effects.append({
            'type': 'hash_modification',
            'method': method
        })
        print("=== Hash Modification Complete ===\n")
        return self

    def relocate_text(self) -> 'VideoTransformer':
        """Detect and relocate text in the video."""
        print("\n=== Starting Text Relocation ===")
        def process_frame(frame, t):
            print(f"\nProcessing text relocation frame at time {t}")
            if not isinstance(frame, np.ndarray):
                print("ERROR: Received non-numpy array frame")
                return frame
                
            frame_copy = frame.copy()
            print(f"Frame shape: {frame_copy.shape}")
            
            try:
                print("Processing frame with TextProcessor")
                inpainted, text_regions = self.text_processor.process_frame(frame_copy, t)
                print(f"Found {len(text_regions)} text regions")
                
                if not text_regions:
                    print("No text regions found")
                    return frame_copy
                    
                safe_positions = self.text_processor.get_safe_positions(inpainted, text_regions)
                print(f"Found {len(safe_positions)} safe positions")
                
                if not safe_positions:
                    print("No safe positions found")
                    return frame_copy
                
                print("Adding text in new positions")
                for region, new_pos in zip(text_regions, safe_positions):
                    new_x, new_y = new_pos
                    cv2.putText(inpainted, region.text, (new_x, new_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
                    cv2.putText(inpainted, region.text, (new_x, new_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                print("Text relocation completed successfully")
                return inpainted
            except Exception as e:
                print(f"ERROR in text relocation: {e}")
                return frame_copy
        
        try:
            # Use fl to get time parameter
            self.transformed_clip = self.video_clip.fl(process_frame)
            print("Text relocation applied to video clip")
            self.effects.append({'type': 'text_relocation'})
        except Exception as e:
            print(f"ERROR applying text relocation: {e}")
        print("=== Text Relocation Complete ===\n")
        return self

    def get_transformed_clip(self) -> VideoFileClip:
        """Get the transformed video clip."""
        print("\n=== Getting Transformed Clip ===")
        if self.transformed_clip is None:
            print("No transformations applied, returning original clip")
            return self.video_clip
        print("Returning transformed clip")
        print("=== Getting Transformed Clip Complete ===\n")
        return self.transformed_clip

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