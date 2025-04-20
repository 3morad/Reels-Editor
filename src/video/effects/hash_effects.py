from moviepy.editor import VideoFileClip
import cv2
import numpy as np
import random
import logging
from scipy.fftpack import dct, idct
from typing import Any
from ..utils.frame_utils import process_frame_safely
from ..utils.logging_utils import configure_logger, timed, log_exceptions

# Configure logger
logger = configure_logger("HashEffects")

# =============================
# Per-frame modifier functions
# =============================

def pixelate_modifier(frame: np.ndarray, intensity: float) -> np.ndarray:
    h, w = frame.shape[:2]
    pattern = random.choice(['dots', 'diagonal', 'grid'])
    if pattern == 'dots':
        step = max(1, int(12 / (intensity + 0.1)))
        dot_size = max(1, int(2 * intensity))
        for y in range(0, h, step):
            for x in range(0, w, step):
                if (x + y) % 4 == 0:
                    y2 = min(y + dot_size, h)
                    x2 = min(x + dot_size, w)
                    ch = random.choice([0,1,2])
                    delta = int(3 + intensity * 10)
                    frame[y:y2, x:x2, ch] = np.clip(frame[y:y2, x:x2, ch] + delta, 0, 255)
    elif pattern == 'diagonal':
        thickness = max(1, int(intensity * 3))
        spacing = max(10, int(30 / (intensity + 0.2)))
        for off in range(0, h+w, spacing):
            ch = random.choice([0,1,2])
            delta = int(5 + intensity * 15)
            for t in range(thickness):
                for i in range(h):
                    j = off - i + t
                    if 0 <= j < w:
                        frame[i, j, ch] = np.clip(frame[i, j, ch] + delta, 0, 255)
    else:
        grid = max(15, int(40 / (intensity + 0.2)))
        lw = max(1, int(intensity * 2))
        delta = int(4 + intensity * 12)
        for y in range(0, h, grid):
            for yy in range(max(0, y-lw//2), min(h, y+lw//2+1)):
                ch = random.choice([0,1,2])
                frame[yy, :, ch] = np.clip(frame[yy, :, ch] + delta, 0, 255)
        for x in range(0, w, grid):
            for xx in range(max(0, x-lw//2), min(w, x+lw//2+1)):
                ch = random.choice([0,1,2])
                frame[:, xx, ch] = np.clip(frame[:, xx, ch] + delta, 0, 255)
    return frame


def glitch_modifier(frame: np.ndarray, intensity: float) -> np.ndarray:
    h, w = frame.shape[:2]
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    gw = max(1, int(w * intensity * 0.01))
    if gw > 0:
        side = random.choice(['left','right'])
        shift = random.randint(-3,3)
        if side == 'left':
            start = random.randint(0, w//4)
        else:
            start = random.randint(3*w//4, w-gw)
        end = min(start + gw, w)
        frame[:, start:end] = np.roll(frame[:, start:end], shift, axis=0)
        ch = 0 if side=='left' else 2
        frame[:, start:end, ch] = np.roll(frame[:, start:end, ch], shift + (2 if side=='left' else -2), axis=0)
    return frame


def dct_modifier(frame: np.ndarray, intensity: float) -> np.ndarray:
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    arr = frame.astype(float)
    for c in range(3):
        mat = arr[:,:,c]
        coeffs = dct(dct(mat.T, norm='ortho').T, norm='ortho')
        cr = int(2 + intensity * 2)
        noise = np.random.normal(0, intensity * 0.5, (cr,cr))
        coeffs[:cr,:cr] += noise
        arr[:,:,c] = idct(idct(coeffs.T, norm='ortho').T, norm='ortho')
    return np.clip(arr, 0, 255).astype(np.uint8)


def noise_modifier(frame: np.ndarray, intensity: float) -> np.ndarray:
    f = frame.astype(np.float32)/255.0
    if random.choice([True,False]):
        noise = np.random.uniform(-0.1,0.1, frame.shape) * intensity
        f = np.clip(f + noise, 0, 1)
    else:
        h,w,_ = frame.shape
        noise = np.random.normal(0, intensity*0.1, (h,w,3))
        cy, cx = random.randint(0,h-1), random.randint(0,w-1)
        rad = random.randint(h//4,h//2)
        yg, xg = np.ogrid[:h,:w]
        mask = np.clip(1 - np.sqrt((yg-cy)**2 + (xg-cx)**2)/rad, 0, 1)[:,:,None]
        f = np.clip(f + noise * mask, 0, 1)
    return (f * 255).astype(np.uint8)


def color_modifier(frame: np.ndarray, intensity: float) -> np.ndarray:
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    h,w,_ = frame.shape
    style = random.choice(['pattern','shift','gradient','selective'])
    if style == 'pattern':
        grid = max(8, int(30/(intensity+0.1)))
        for i in range(0,h,grid):
            for j in range(0,w,grid):
                if ((i//grid)+(j//grid))%2 == 0:
                    c = random.randint(0,2)
                    v = random.randint(2, int(5+intensity*15))
                    frame[i:i+grid, j:j+grid, c] = np.clip(frame[i:i+grid, j:j+grid, c] + v, 0, 255)
    elif style == 'shift':
        shifts = [random.uniform(-intensity*8, intensity*8) for _ in range(3)]
        avg = sum(shifts)/3
        for c in range(3):
            frame[:,:,c] = np.clip(frame[:,:,c] + (shifts[c]-avg), 0, 255)
    elif style == 'gradient':
        dir = random.choice(['horizontal','vertical','radial'])
        if dir == 'horizontal':
            for j in range(w):
                f = j/w
                for c in range(3):
                    frame[:,j,c] = np.clip(frame[:,j,c] + random.uniform(-1,1)*intensity*10*f, 0, 255)
        elif dir == 'vertical':
            for i in range(h):
                f = i/h
                for c in range(3):
                    frame[i,:,c] = np.clip(frame[i,:,c] + random.uniform(-1,1)*intensity*10*f, 0, 255)
        else:
            cy, cx = h//2, w//2
            maxd = np.sqrt(cy**2 + cx**2)
            for i in range(h):
                for j in range(w):
                    f = np.sqrt((i-cy)**2 + (j-cx)**2)/maxd
                    for c in range(3):
                        frame[i,j,c] = np.clip(frame[i,j,c] + random.uniform(-1,1)*intensity*10*f, 0, 255)
    else:
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        masks = {
            'red': cv2.bitwise_or(cv2.inRange(hsv,(0,50,50),(10,255,255)), cv2.inRange(hsv,(170,50,50),(180,255,255))),
            'green': cv2.inRange(hsv,(40,50,50),(80,255,255)),
            'blue': cv2.inRange(hsv,(100,50,50),(140,255,255)),
            'yellow': cv2.inRange(hsv,(20,100,100),(30,255,255)),
            'cyan': cv2.inRange(hsv,(85,50,50),(95,255,255)),
            'magenta': cv2.inRange(hsv,(140,50,50),(160,255,255))
        }
        choice = random.choice(list(masks))
        mask3 = masks[choice][:,:,None]/255.0
        adj = np.random.uniform(-intensity*15, intensity*15, (1,3))
        frame = np.clip(frame + adj * mask3, 0, 255).astype(np.uint8)
    return frame


def watermark_modifier(frame: np.ndarray, intensity: float) -> np.ndarray:
    h, w = frame.shape[:2]
    size = min(32, h//8, w//8)
    wm = np.zeros((size, size, 3), dtype=np.float32)
    cy, cx = size//2, size//2
    ry = size//4
    yy, xx = np.ogrid[:size, :size]
    mask = (yy-cy)**2 + (xx-cx)**2 <= ry**2
    wm[mask] = [intensity*0.8, intensity*0.7, intensity*0.9]
    pos = random.choice(['tl','tr','bl','br','rnd'])
    if pos == 'tl': x0,y0 = 10,10
    elif pos == 'tr': x0,y0 = w-size-10,10
    elif pos == 'bl': x0,y0 = 10,h-size-10
    elif pos == 'br': x0,y0 = w-size-10,h-size-10
    else: x0,y0 = random.randint(0,w-size), random.randint(0,h-size)
    roi = frame[y0:y0+size, x0:x0+size].astype(np.float32)
    strength = 35 + intensity*15
    roi = np.clip(roi + wm*strength, 0, 255)
    frame[y0:y0+size, x0:x0+size] = roi.astype(np.uint8)
    return frame

# =============================
# Clip-level modifier functions
# =============================

@timed
@log_exceptions
def _apply_delay_modification(clip: VideoFileClip, intensity: float) -> VideoFileClip:
    return clip.set_start((10 + intensity*50)/1000.0)

@timed
@log_exceptions
def _apply_metadata_modification(clip: VideoFileClip, intensity: float) -> VideoFileClip:
    clip = clip.set_fps(clip.fps + random.uniform(-0.5,0.5)*intensity)
    scale = 1 + random.uniform(-0.05,0.05)*intensity
    clip = clip.resize(width=int(clip.w*scale), height=int(clip.h*scale))
    clip = clip.set_duration(clip.duration + random.uniform(-0.01,0.01)*intensity)
    rot = random.uniform(-0.5,0.5)*intensity
    if abs(rot) > 0.01:
        clip = clip.rotate(rot)
    return clip

@timed
@log_exceptions
def _apply_temporal_modification(clip: VideoFileClip, intensity: float) -> VideoFileClip:
    return clip  # no-op for now

# =============================
# Dispatcher entrypoints
# =============================

@timed
@log_exceptions
def process_hash_frame(frame: Any, hash_type: str, intensity: float) -> Any:
    return process_frame_safely(frame, lambda f: globals()[f"{hash_type}_modifier"](f, intensity))

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