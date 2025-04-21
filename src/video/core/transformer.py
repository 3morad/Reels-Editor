from moviepy.editor import VideoFileClip, ImageSequenceClip, vfx
import os
import logging
import time
import random
import numpy as np
from typing import Optional, List, Dict, Any, Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial

# Try importing GPU libraries with graceful fallbacks
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..effects import basic_effects, hash_effects, gpu_effects
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
    def __init__(self, video_clip: VideoFileClip):
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

        # Queues of transforms
        self._frame_funcs: List[Callable[[np.ndarray], np.ndarray]] = []
        self._clip_funcs: List[Callable[[VideoFileClip], VideoFileClip]] = []
        self.effects: List[Dict[str, Any]] = []
        self.transformed_clip: Optional[VideoFileClip] = None
        
        # GPU settings
        self.use_gpu = gpu_effects.gpu_effects_available()
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
        
        # Clean up GPU resources
        if self.use_gpu:
            gpu_manager.cleanup()

    def apply_zoom(self, zoom_factor: float = 1.2) -> 'VideoTransformer':
        logger.debug(f"Registering zoom: factor={zoom_factor}")
        if self.use_gpu:
            self._frame_funcs.append(partial(gpu_effects.process_zoom_frame, zoom_factor=zoom_factor))
        else:
            self._frame_funcs.append(partial(basic_effects.process_zoom_frame, zoom_factor=zoom_factor))
        self.effects.append({'type':'zoom','zoom_factor':zoom_factor})
        return self

    def apply_crop(self, crop_percent: float = 0.1) -> 'VideoTransformer':
        logger.debug(f"Registering crop: percent={crop_percent}")
        if self.use_gpu:
            self._frame_funcs.append(partial(gpu_effects.process_crop_frame, crop_percent=crop_percent))
        else:
            self._frame_funcs.append(partial(basic_effects.process_crop_frame, crop_percent=crop_percent))
        self.effects.append({'type':'crop','crop_percent':crop_percent})
        return self

    def apply_filter(self, filter_type: str, intensity: float = 1.0) -> 'VideoTransformer':
        logger.debug(f"Registering filter: type={filter_type}, intensity={intensity}")
        if self.use_gpu:
            self._frame_funcs.append(partial(gpu_effects.process_filter_frame, filter_type=filter_type, intensity=intensity))
        else:
            self._frame_funcs.append(partial(basic_effects.process_filter_frame, filter_type=filter_type, intensity=intensity))
        self.effects.append({'type':'filter','filter_type':filter_type,'intensity':intensity})
        return self

    def apply_transition(self, transition_type: str, duration: float = 1.0) -> 'VideoTransformer':
        logger.debug(f"Registering transition: {transition_type}({duration})")
        self._clip_funcs.append(partial(_transition_clip, transition_type=transition_type, duration=duration))
        self.effects.append({'type':'transition','transition_type':transition_type,'duration':duration})
        return self

    def apply_trim(self, trim_percent: float = 0.1) -> 'VideoTransformer':
        logger.debug(f"Registering trim: percent={trim_percent}")
        self._clip_funcs.append(partial(_trim_clip, trim_percent=trim_percent))
        self.effects.append({'type':'trim','trim_percent':trim_percent})
        return self

    def modify_hash(self, hash_type: str, intensity: float = 1.0) -> 'VideoTransformer':
        logger.debug(f"Registering hash modification: type={hash_type}, intensity={intensity}")
        per_frame = {'pixelate','glitch','dct','noise','color','watermark'}
        clip_level = {'metadata','delay','temporal'}
        if hash_type in per_frame:
            self._frame_funcs.append(partial(hash_effects.process_hash_frame_gpu, hash_type=hash_type, intensity=intensity))
        elif hash_type in clip_level:
            func_map = {
                'metadata': hash_effects.process_metadata_clip,
                'delay':    hash_effects.process_delay_clip,
                'temporal': hash_effects.process_temporal_clip
            }
            self._clip_funcs.append(partial(func_map[hash_type], intensity=intensity))
        else:
            logger.warning(f"Unknown hash_type: {hash_type}")
        self.effects.append({'type':hash_type,'intensity':intensity})
        return self

    def bake_gpu(self) -> VideoFileClip:
        """Process frames using GPU batch processing for efficiency."""
        if not TORCH_AVAILABLE or not gpu_manager.has_cuda:
            logger.warning("GPU processing requested but not available, falling back to CPU")
            return self.bake_parallel()
            
        fps = self.video_clip.fps
        
        # Get video dimensions for batch size calculation
        height, width = self.video_clip.h, self.video_clip.w
        
        # Calculate optimal batch size based on GPU memory
        batch_size = gpu_manager.get_optimal_batch_size(height, width)
        logger.info(f"Using GPU batch size of {batch_size} for {width}x{height} frames")
        
        # Extract all frames first to avoid memory issues with random frame access
        logger.info("Extracting frames for GPU processing")
        all_frames = list(self.video_clip.iter_frames(fps=fps, dtype='uint8'))
        total_frames = len(all_frames)
        logger.info(f"Processing {total_frames} frames with {len(self._frame_funcs)} frame funcs")
        
        # Process in batches
        processed_frames = []
        
        for i in range(0, total_frames, batch_size):
            batch_end = min(i + batch_size, total_frames)
            logger.debug(f"Processing GPU batch {i//batch_size + 1}/{(total_frames + batch_size - 1)//batch_size}")
            
            # Process frames in current batch with GPU
            batch_frames = all_frames[i:batch_end]
            
            # Convert to tensors for GPU processing
            try:
                # Process frames sequentially using GPU
                batch_processed = []
                for frame in batch_frames:
                    # Apply each frame function
                    result = frame
                    for fn in self._frame_funcs:
                        try:
                            result = fn(result)
                        except Exception as e:
                            logger.warning(f"Error in {getattr(fn, '__name__', repr(fn))}: {e}")
                            # If the function has a fallback CPU implementation included in it,
                            # it should handle the fallback internally. Otherwise, just pass through.
                            pass
                    batch_processed.append(result)
                
                # Add processed batch to results
                processed_frames.extend(batch_processed)
                
                # Force GPU cleanup
                if TORCH_AVAILABLE and gpu_manager.has_cuda:
                    gpu_manager.cleanup()
                    
            except Exception as e:
                logger.error(f"Error in GPU batch processing: {e}")
                logger.warning("Falling back to CPU for this batch")
                
                # Process this batch on CPU
                batch_processed = []
                for frame in batch_frames:
                    # Apply frame functions on CPU
                    result = frame
                    for fn in self._frame_funcs:
                        try:
                            # Check if this is a partial function
                            if isinstance(fn, partial):
                                # Create a CPU version with the same parameters
                                fn_name = fn.func.__name__
                                # If this is a GPU effect, find equivalent CPU version
                                if 'gpu_effects' in str(fn.func):
                                    cpu_fn_name = fn_name.replace('process_', '')
                                    cpu_fn = getattr(basic_effects, f"process_{cpu_fn_name}", None)
                                    if cpu_fn:
                                        result = cpu_fn(result, **fn.keywords)
                                    else:
                                        result = fn(result)
                                else:
                                    result = fn(result)
                            else:
                                result = fn(result)
                        except Exception as inner_e:
                            logger.warning(f"CPU fallback error in {getattr(fn, '__name__', repr(fn))}: {inner_e}")
                            # Just pass through if CPU also fails
                            pass
                    batch_processed.append(result)
                
                processed_frames.extend(batch_processed)
                
            # Log progress
            logger.debug(f"Processed {batch_end}/{total_frames} frames ({batch_end/total_frames*100:.1f}%)")
        
        logger.info("GPU frame processing complete, building clip")
        
        # Build clip from processed frames
        clip = ImageSequenceClip(processed_frames, fps=fps)
        
        # Apply clip-level effects
        for fn in self._clip_funcs:
            try:
                # Safely get function name for logging
                if isinstance(fn, partial):
                    name = fn.func.__name__
                else:
                    name = getattr(fn, '__name__', repr(fn))
                logger.info(f"Applying clip func: {name}")
                clip = fn(clip)
            except Exception as e:
                logger.warning(f"Error applying clip effect {name}: {e}")
                # Continue with other effects
            
        self.transformed_clip = clip
        logger.info("GPU bake complete")
        return clip

    def bake_parallel(self, workers: Optional[int] = None) -> VideoFileClip:
        fps = self.video_clip.fps
        frames = list(self.video_clip.iter_frames(fps=fps, dtype='uint8'))
        logger.info(f"Baking {len(frames)} frames with {len(self._frame_funcs)} frame funcs")
        max_workers = workers or max(1, os.cpu_count()-1)
        with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker, initargs=(self._frame_funcs,)) as pool:
            processed = list(pool.map(_apply_global_frame_funcs, frames))
        logger.info("Frame processing complete, building clip")
        clip = ImageSequenceClip(processed, fps=fps)
        for fn in self._clip_funcs:
            # Safely get function name for logging
            if isinstance(fn, partial):
                name = fn.func.__name__
            else:
                name = getattr(fn, '__name__', repr(fn))
            logger.info(f"Applying clip func: {name}")
            clip = fn(clip)
        self.transformed_clip = clip
        logger.info("Parallel bake complete")
        return clip

    def get_transformed_clip(self) -> VideoFileClip:
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
