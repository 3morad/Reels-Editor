from moviepy.editor import VideoFileClip, ImageSequenceClip, vfx
import os
import logging
import time
import random
import numpy as np
from typing import Optional, List, Dict, Any, Callable
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from ..effects import basic_effects, hash_effects
from ..utils.logging_utils import configure_logger

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
            scale = min(1080 / video_clip.h, 1920 / video_clip.w)
            new_h = int(video_clip.h * scale)
            new_w = int(video_clip.w * scale)
            logger.info(f"Resizing for speed: {new_w}x{new_h}")
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

    def apply_zoom(self, zoom_factor: float = 1.2) -> 'VideoTransformer':
        logger.debug(f"Registering zoom: factor={zoom_factor}")
        self._frame_funcs.append(partial(basic_effects.process_zoom_frame, zoom_factor=zoom_factor))
        self.effects.append({'type':'zoom','zoom_factor':zoom_factor})
        return self

    def apply_crop(self, crop_percent: float = 0.1) -> 'VideoTransformer':
        logger.debug(f"Registering crop: percent={crop_percent}")
        self._frame_funcs.append(partial(basic_effects.process_crop_frame, crop_percent=crop_percent))
        self.effects.append({'type':'crop','crop_percent':crop_percent})
        return self

    def apply_filter(self, filter_type: str, intensity: float = 1.0) -> 'VideoTransformer':
        logger.debug(f"Registering filter: type={filter_type}, intensity={intensity}")
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
            self._frame_funcs.append(partial(hash_effects.process_hash_frame, hash_type=hash_type, intensity=intensity))
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
            return self.bake_parallel()
        return self.transformed_clip

    def get_effects(self) -> List[Dict[str,Any]]:
        return self.effects

    def reset(self) -> 'VideoTransformer':
        self.cleanup()
        return self
