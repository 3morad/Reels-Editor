import torch
import torch.nn.functional as F
import numpy as np
import random
from typing import Any
from moviepy.editor import VideoFileClip
from ..utils.logging_utils import configure_logger, timed, log_exceptions

logger = configure_logger("GPUHashEffects")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================
# GPU-optimized modifier functions
# =============================

@torch.no_grad()
def pixelate_modifier_gpu(tensor: torch.Tensor, intensity: float) -> torch.Tensor:
    _, _, h, w = tensor.shape
    step = max(1, int(12 / (intensity + 0.1)))
    dot_size = max(1, int(2 * intensity))

    mask = torch.zeros_like(tensor)

    for y in range(0, h, step):
        for x in range(0, w, step):
            if (x + y) % 4 == 0:
                y2 = min(y + dot_size, h)
                x2 = min(x + dot_size, w)
                ch = random.randint(0, 2)
                delta = int(3 + intensity * 10)
                mask[:, ch, y:y2, x:x2] += delta

    tensor = torch.clamp(tensor + mask, 0, 255)
    return tensor

@torch.no_grad()
def glitch_modifier_gpu(tensor: torch.Tensor, intensity: float) -> torch.Tensor:
    _, _, h, w = tensor.shape
    gw = max(1, int(w * intensity * 0.01))
    if gw > 0:
        side = random.choice(['left', 'right'])
        shift = random.randint(-3, 3)
        start = random.randint(0, w // 4) if side == 'left' else random.randint(3 * w // 4, w - gw)
        end = min(start + gw, w)

        tensor[:, :, :, start:end] = torch.roll(tensor[:, :, :, start:end], shifts=shift, dims=2)
        ch = 0 if side == 'left' else 2
        tensor[:, ch, :, start:end] = torch.roll(tensor[:, ch, :, start:end], shifts=shift + (2 if side == 'left' else -2), dims=1)

    return tensor

@torch.no_grad()
def noise_modifier_gpu(tensor: torch.Tensor, intensity: float) -> torch.Tensor:
    noise = torch.randn_like(tensor) * intensity * 25.5
    tensor = torch.clamp(tensor + noise, 0, 255)
    return tensor

@torch.no_grad()
def color_modifier_gpu(tensor: torch.Tensor, intensity: float) -> torch.Tensor:
    shifts = torch.tensor([random.uniform(-intensity * 8, intensity * 8) for _ in range(3)], device=device).view(1,3,1,1)
    tensor = torch.clamp(tensor + shifts, 0, 255)
    return tensor

@torch.no_grad()
def watermark_modifier_gpu(tensor: torch.Tensor, intensity: float) -> torch.Tensor:
    _, _, h, w = tensor.shape
    size = min(32, h // 8, w // 8)
    wm = torch.zeros((1, 3, size, size), device=device)

    cy, cx, ry = size // 2, size // 2, size // 4
    yy, xx = torch.meshgrid(torch.arange(size, device=device), torch.arange(size, device=device), indexing='ij')
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= ry ** 2
    wm[:, :, mask] = torch.tensor([intensity * 0.8, intensity * 0.7, intensity * 0.9], device=device).view(1,3,1)

    x0, y0 = random.randint(0, w - size), random.randint(0, h - size)
    strength = 35 + intensity * 15

    tensor[:, :, y0:y0+size, x0:x0+size] = torch.clamp(
        tensor[:, :, y0:y0+size, x0:x0+size] + wm * strength, 0, 255
    )

    return tensor

modifier_funcs_gpu = {
    'pixelate': pixelate_modifier_gpu,
    'glitch': glitch_modifier_gpu,
    'noise': noise_modifier_gpu,
    'color': color_modifier_gpu,
    'watermark': watermark_modifier_gpu,
}

# =============================
# Clip-level modifier functions
# =============================

@timed
@log_exceptions
def _apply_delay_modification(clip: VideoFileClip, intensity: float) -> VideoFileClip:
    return clip.set_start((10 + intensity * 50) / 1000.0)

@timed
@log_exceptions
def _apply_metadata_modification(clip: VideoFileClip, intensity: float) -> VideoFileClip:
    clip = clip.set_fps(clip.fps + random.uniform(-0.5, 0.5) * intensity)
    scale = 1 + random.uniform(-0.05, 0.05) * intensity
    clip = clip.resize(width=int(clip.w * scale), height=int(clip.h * scale))
    clip = clip.set_duration(clip.duration + random.uniform(-0.01, 0.01) * intensity)
    rot = random.uniform(-0.5, 0.5) * intensity
    if abs(rot) > 0.01:
        clip = clip.rotate(rot)
    return clip

@timed
@log_exceptions
def _apply_temporal_modification(clip: VideoFileClip, intensity: float) -> VideoFileClip:
    return clip  # no-op for now

# =============================
# GPU Dispatcher entrypoints
# =============================

@timed
@log_exceptions
def process_hash_frame_gpu(frame: np.ndarray, hash_type: str, intensity: float) -> np.ndarray:
    if hash_type not in modifier_funcs_gpu:
        raise ValueError(f"Unknown GPU hash modifier: {hash_type}")

    tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().to(device)
    tensor = modifier_funcs_gpu[hash_type](tensor, intensity)
    result = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return result

@timed
@log_exceptions
def process_delay_clip(clip: VideoFileClip, intensity: float) -> VideoFileClip:
    return _apply_delay_modification(clip, intensity)

@timed
@log_exceptions
def process_metadata_clip(clip: VideoFileClip, intensity: float) -> VideoFileClip:
    return _apply_metadata_modification(clip, intensity)

@timed
@log_exceptions
def process_temporal_clip(clip: VideoFileClip, intensity: float) -> VideoFileClip:
    return _apply_temporal_modification(clip, intensity)
