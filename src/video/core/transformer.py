from moviepy.editor import VideoFileClip, ImageSequenceClip, vfx
import os
import logging
import time
import random
import numpy as np
import subprocess
import tempfile
from typing import Optional, List, Dict, Any, Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import uuid

# Try importing GPU libraries with graceful fallbacks
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..effects import ffmpeg_basic_effects, ffmpeg_hash_effects
from ..utils.logging_utils import configure_logger
from ..utils.gpu_manager import gpu_manager

# Configure logger
logger = configure_logger("VideoTransformer")

# Global frame funcs storage for worker processes
_GLOBAL_FRAME_FUNCS: List[Callable[[np.ndarray], np.ndarray]] = []

def _init_worker(frame_funcs):
    """Initializer for ProcessPoolExecutor to set global frame funcs."""
    global _GLOBAL_FRAME_FUNCS
    _GLOBAL_FRAME_FUNCS = frame_funcs
    logger.debug(f"Worker initialized with {len(frame_funcs)} frame funcs")


def _apply_global_frame_funcs(frame: np.ndarray) -> np.ndarray:
    """Module-level helper to apply all registered frame funcs."""
    out = frame
    for fn in _GLOBAL_FRAME_FUNCS:
        before = out.copy()
        out = fn(out)
        # Safely get function name
        if isinstance(fn, partial):
            name = fn.func.__name__
        else:
            name = getattr(fn, '__name__', repr(fn))
        logger.debug(f"Applied {name} to frame: pixel[0,0] {before[0,0]} -> {out[0,0]}")
    return out


def _transition_clip(clip: VideoFileClip, transition_type: str, duration: float) -> VideoFileClip:
    """Apply fadein/fadeout transitions at clip level using vfx."""
    logger.debug(f"Applying clip-level transition: {transition_type}({duration})")
    func = getattr(vfx, transition_type)
    return func(clip, duration)


def _trim_clip(clip: VideoFileClip, trim_percent: float) -> VideoFileClip:
    """Trim end of clip by a percentage."""
    logger.debug(f"Applying clip-level trim: {trim_percent*100:.1f}% end")
    end_t = clip.duration * (1 - trim_percent)
    return clip.subclip(0, end_t)

class VideoTransformer:
    def __init__(self, video_clip: VideoFileClip, input_path: Optional[str] = None):
        """Initialize transformer; register per-frame and clip-level effects."""
        self.start_time = time.time()
        logger.info("=== VideoTransformer Initialization ===")
        if video_clip is None:
            raise ValueError("Video clip cannot be None")
        if not hasattr(video_clip, 'h') or not hasattr(video_clip, 'w'):
            raise ValueError("Video clip must have height and width")

        # Downscale for performance if high resolution
        if video_clip.h > 1080 or video_clip.w > 1920:
            # Ensure even dimensions for better compatibility with video codecs
            scale = min(1080 / video_clip.h, 1920 / video_clip.w)
            new_h = int(video_clip.h * scale)
            new_w = int(video_clip.w * scale)
            
            # Ensure dimensions are even (required by many codecs)
            new_h = new_h if new_h % 2 == 0 else new_h - 1
            new_w = new_w if new_w % 2 == 0 else new_w - 1
            
            logger.info(f"Resizing for speed: {new_w}x{new_h}")
            self.video_clip = video_clip.resize(width=new_w, height=new_h)
            self.original_clip = video_clip
        else:
            # Still ensure even dimensions for the original clip
            new_h = video_clip.h if video_clip.h % 2 == 0 else video_clip.h - 1
            new_w = video_clip.w if video_clip.w % 2 == 0 else video_clip.w - 1
            
            if new_h != video_clip.h or new_w != video_clip.w:
                logger.info(f"Adjusting to even dimensions: {new_w}x{new_h}")
                self.video_clip = video_clip.resize(width=new_w, height=new_h)
                self.original_clip = video_clip
            else:
                self.video_clip = video_clip
                self.original_clip = None

        # Store the original input file path if provided
        self.input_path = input_path

        # Queues of transforms
        self._frame_funcs: List[Callable[[np.ndarray], np.ndarray]] = []
        self._clip_funcs: List[Callable[[VideoFileClip], VideoFileClip]] = []
        self.effects: List[Dict[str, Any]] = []
        self.transformed_clip: Optional[VideoFileClip] = None
        
        # FFmpeg command parameters
        self.ffmpeg_params: List[str] = []
        
        # GPU settings
        self.use_gpu = TORCH_AVAILABLE and torch.cuda.is_available()
        if self.use_gpu:
            logger.info("GPU acceleration enabled for frame processing")
        else:
            logger.info("GPU acceleration not available, using CPU")
            
        logger.info(f"Initialized in {time.time() - self.start_time:.2f}s")

    def cleanup(self):
        """Close transformed clip and reset state."""
        logger.debug("Cleaning up VideoTransformer resources")
        if self.transformed_clip and self.transformed_clip != self.video_clip:
            try: self.transformed_clip.close()
            except: pass
        if self.original_clip:
            try: self.original_clip.close()
            except: pass
        self.transformed_clip = None
        self.effects.clear()
        self._frame_funcs.clear()
        self._clip_funcs.clear()
        self.ffmpeg_params.clear()
        
        # Clean up GPU resources
        if self.use_gpu:
            gpu_manager.cleanup()

    def apply_crop(self, crop_percent: float = None) -> 'VideoTransformer':
        # Subtle crop: 1-3%
        crop_percent = crop_percent if crop_percent is not None else random.uniform(0.01, 0.03)
        logger.debug(f"Registering crop: percent={crop_percent}")
        self.ffmpeg_params.extend(ffmpeg_basic_effects.get_crop_params(crop_percent))
        self.effects.append({'type':'crop','crop_percent':crop_percent})
        return self

    def apply_zoom(self, zoom_factor: float = None) -> 'VideoTransformer':
        # Subtle zoom: 1.01-1.05
        zoom_factor = zoom_factor if zoom_factor is not None else random.uniform(1.01, 1.05)
        logger.debug(f"Registering zoom: factor={zoom_factor}")
        self.ffmpeg_params.extend(ffmpeg_basic_effects.get_zoom_params(zoom_factor))
        self.effects.append({'type':'zoom','zoom_factor':zoom_factor})
        return self

    def apply_filter(self, filter_type: str = None, intensity: float = None) -> 'VideoTransformer':
        # Only brightness or contrast, small random intensity
        filter_type = filter_type if filter_type in ['brightness', 'contrast'] else random.choice(['brightness', 'contrast'])
        # Neutral is 0.0 for brightness, 1.0 for contrast; we want a small deviation
        if filter_type == 'brightness':
            intensity = intensity if intensity is not None else random.uniform(0.45, 0.55)  # 0.5 is neutral
        else:
            intensity = intensity if intensity is not None else random.uniform(0.95, 1.05)  # 1.0 is neutral
        logger.debug(f"Registering filter: type={filter_type}, intensity={intensity}")
        self.ffmpeg_params.extend(ffmpeg_basic_effects.get_filter_params(filter_type, intensity))
        self.effects.append({'type':'filter','filter_type':filter_type,'intensity':intensity})
        return self

    def apply_transition(self, transition_type: str = None, duration: float = None) -> 'VideoTransformer':
        # Fadein or fadeout, 0.1-0.5s
        transition_type = transition_type if transition_type in ['fadein', 'fadeout'] else random.choice(['fadein', 'fadeout'])
        duration = duration if duration is not None else random.uniform(0.1, 0.5)
        # Use trimmed duration if trim is applied, else use original
        video_duration = None
        for param in self.ffmpeg_params:
            if param == '-t':
                try:
                    video_duration = float(self.ffmpeg_params[self.ffmpeg_params.index(param)+1])
                except Exception:
                    pass
        if video_duration is None and hasattr(self.video_clip, 'duration'):
            video_duration = self.video_clip.duration
        logger.debug(f"Registering transition: {transition_type}({duration}), video_duration={video_duration}")
        self.ffmpeg_params.extend(ffmpeg_basic_effects.get_transition_params(transition_type, duration, video_duration=video_duration))
        self.effects.append({'type':'transition','transition_type':transition_type,'duration':duration})
        return self

    def apply_trim(self, trim_percent: float = None) -> 'VideoTransformer':
        # Trim 5-10% from end
        trim_percent = trim_percent if trim_percent is not None else random.uniform(0.05, 0.1)
        logger.debug(f"Registering trim: percent={trim_percent}")
        duration = self.video_clip.duration if hasattr(self.video_clip, 'duration') else None
        self.ffmpeg_params.extend(ffmpeg_basic_effects.get_trim_params(trim_percent, duration=duration))
        self.effects.append({'type':'trim','trim_percent':trim_percent})
        return self

    def modify_hash(self, hash_type: str, intensity: float = 1.0, preset: str = None) -> 'VideoTransformer':
        """Apply a single hash modification effect."""
        logger.debug(f"Registering hash modification: type={hash_type}, intensity={intensity}, preset={preset}")
        
        # If preset is provided, use its default intensity if not explicitly overridden
        if preset and intensity == 1.0:  # Only use preset intensity if not explicitly set
            try:
                from ..effects.hash_presets import get_preset_default_intensity
                intensity = get_preset_default_intensity(preset, hash_type)
                logger.debug(f"Using preset intensity {intensity} for {hash_type} from preset {preset}")
            except ValueError as e:
                logger.warning(f"Could not get preset intensity: {e}. Using default intensity.")
        
        self.ffmpeg_params.extend(ffmpeg_hash_effects.get_hash_params(hash_type, intensity))
        self.effects.append({'type': hash_type, 'intensity': intensity, 'preset': preset})
        return self

    def apply_hash_preset(self, preset: str, methods: List[str] = None) -> 'VideoTransformer':
        """Apply all hash modifications from a preset.
        
        Args:
            preset: The preset to use ('fast', 'normal', or 'slow')
            methods: Optional list of specific methods to apply. If None, applies all methods from preset.
        """
        logger.debug(f"Applying hash preset: {preset} with methods: {methods}")
        
        try:
            from ..effects.hash_presets import get_preset_methods, get_preset_default_intensity
            
            # Get all methods for this preset
            preset_methods = get_preset_methods(preset)
            
            # If specific methods requested, filter to only those methods
            if methods:
                preset_methods = [m for m in preset_methods if m in methods]
                if not preset_methods:
                    logger.warning(f"None of the requested methods {methods} found in preset {preset}")
                    return self
            
            # Apply each method with its preset intensity
            for method in preset_methods:
                try:
                    intensity = get_preset_default_intensity(preset, method)
                    self.modify_hash(method, intensity, preset)
                except ValueError as e:
                    logger.warning(f"Skipping method {method}: {e}")
                    continue
                
        except ValueError as e:
            logger.error(f"Error applying preset {preset}: {e}")
            
        return self

    def bake_gpu(self) -> VideoFileClip:
        """Process frames using GPU batch processing for efficiency."""
        # For FFmpeg implementation, we'll use the same method as bake_parallel
        # since FFmpeg handles the processing internally
        return self.bake_parallel()

    def bake_parallel(self, workers: Optional[int] = None) -> str:
        """Process video using FFmpeg for effects and return output file path (no MoviePy reload)."""
        logger.info("Processing video with FFmpeg effects")
        
        # Use original input file if no MoviePy effects are queued
        use_original_input = (
            self.input_path is not None and
            not self._frame_funcs and
            not self._clip_funcs
        )
        if use_original_input:
            input_path = self.input_path
        else:
            # Create temporary input file using MoviePy
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as input_temp:
                input_path = input_temp.name
                self.video_clip.write_videofile(input_path, codec='libx264', audio=False, 
                                               preset='ultrafast', threads=workers or os.cpu_count())
        
        # Create a unique output path to prevent file conflicts between processes
        unique_id = str(uuid.uuid4())[:8]  # Use first 8 chars of UUID for brevity
        output_path = os.path.join(os.path.dirname(input_path), f"output_{int(time.time())}_{unique_id}.mp4")
        
        # Build FFmpeg command with saved effects
        success = False
        attempts = 0
        max_attempts = 2  # Try twice with original effects, then fallback
        
        while not success and attempts < max_attempts + 1:
            try:
                # Build FFmpeg command
                cmd = ['ffmpeg', '-y', '-i', input_path]
                
                if attempts < max_attempts:
                    # Use original effects for first attempts
                    # Collect and combine video filters
                    video_filters = []
                    other_params = []
                    for param in self.ffmpeg_params:
                        if param == '-vf':
                            continue
                        elif isinstance(param, str) and '=' in param:
                            video_filters.append(param)
                        else:
                            other_params.append(param)
                    
                    # Combine all video filters into a single filter chain
                    if video_filters:
                        # Always ensure even dimensions at the end for codec compatibility
                        video_filters.append("scale='if(mod(iw,2),iw-1,iw)':'if(mod(ih,2),ih-1,ih)'")
                        video_filters.append("format=yuv420p")
                        cmd.extend(['-vf', ','.join(video_filters)])
                    
                    # Add any other parameters
                    cmd.extend(other_params)
                else:
                    # Last attempt: use guaranteed fallback effect
                    from ..effects.ffmpeg_hash_effects import get_fallback_params
                    fallback_params = get_fallback_params()
                    
                    # Extract the filter part
                    if fallback_params and fallback_params[0] == '-vf':
                        filter_value = fallback_params[1]
                        # Add compatibility filters
                        filters = [
                            filter_value,
                            "scale='if(mod(iw,2),iw-1,iw)':'if(mod(ih,2),ih-1,ih)'",
                            "format=yuv420p"
                        ]
                        cmd.extend(['-vf', ','.join(filters)])
                    logger.warning("Using fallback effects after previous failures")
                
                # Explicitly disable audio and only map video stream
                cmd.extend(['-an', '-map', '0:v:0'])
                
                # Add output path
                cmd.append(output_path)
                
                # Execute FFmpeg command
                logger.info(f"Running FFmpeg command (attempt {attempts+1}): {' '.join(cmd)}")
                result = subprocess.run(cmd, check=True, capture_output=True)
                logger.info(f"FFmpeg processing completed successfully (return code: {result.returncode})")
                
                # Verify output file exists
                if not os.path.exists(output_path):
                    logger.error(f"FFmpeg output file doesn't exist: {output_path}")
                    logger.error(f"FFmpeg stdout: {result.stdout.decode() if result.stdout else 'None'}")
                    logger.error(f"FFmpeg stderr: {result.stderr.decode() if result.stderr else 'None'}")
                    raise RuntimeError(f"FFmpeg completed but output file {output_path} not found")
                    
                # Verify output file has content
                file_size = os.path.getsize(output_path)
                if file_size < 1000:  # Less than 1KB is suspicious
                    logger.warning(f"FFmpeg output file is very small: {file_size} bytes")
                    if attempts < max_attempts:
                        raise RuntimeError("Output file too small, trying again")
                
                success = True  # If we get here, all checks passed
                
            except Exception as e:
                attempts += 1
                error_msg = str(e)
                if isinstance(e, subprocess.CalledProcessError) and e.stderr:
                    error_msg = e.stderr.decode() if hasattr(e.stderr, 'decode') else str(e.stderr)
                
                if attempts <= max_attempts:
                    logger.warning(f"FFmpeg attempt {attempts} failed: {error_msg}")
                    # Generate a new unique output path for the next attempt
                    unique_id = str(uuid.uuid4())[:8]
                    output_path = os.path.join(os.path.dirname(input_path), f"output_{int(time.time())}_{unique_id}.mp4")
                else:
                    logger.error(f"All FFmpeg attempts failed: {error_msg}")
                    raise RuntimeError(f"FFmpeg processing failed after {attempts} attempts: {error_msg}")
        
        # Clean up temporary input file if we created it
        if not use_original_input:
            try:
                os.unlink(input_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary input file: {e}")
        
        # Return the output file path directly (no MoviePy reload)
        return output_path

    def get_transformed_clip(self) -> str:
        if self.transformed_clip is None:
            if self.use_gpu and gpu_manager.should_use_gpu():
                return self.bake_gpu()
            else:
                return self.bake_parallel()
        return self.transformed_clip

    def get_effects(self) -> List[Dict[str,Any]]:
        return self.effects

    def reset(self) -> 'VideoTransformer':
        self.cleanup()
        return self
