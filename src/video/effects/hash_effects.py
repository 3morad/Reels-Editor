from moviepy.editor import VideoFileClip
import cv2
import numpy as np
import random
import time
import logging
import traceback
from scipy.fftpack import dct, idct
from typing import Optional, Union, Tuple
from ..utils.frame_utils import validate_frame, process_frame_safely
from ..utils.logging_utils import configure_logger, timed, log_exceptions

# Configure logger
logger = configure_logger("HashEffects")

@timed
@log_exceptions
def modify_hash(clip: VideoFileClip, hash_type: str, intensity: float = 1.0) -> VideoFileClip:
    """Apply hash effect to the video clip"""
    if not isinstance(clip, VideoFileClip):
        raise ValueError("Input must be a VideoFileClip")
        
    if not 0 <= intensity <= 1.0:
        raise ValueError("Intensity must be between 0 and 1")
        
    logger.info(f"=== Starting {hash_type} hash modification ===")
    logger.info(f"Input clip properties: {clip.size}, {clip.duration}s, {clip.fps}fps")
    logger.info(f"Intensity: {intensity}")
    
    # Map of available hash types to their modification functions
    hash_functions = {
        "pixelate": _apply_pixel_modification,
        "glitch": _apply_glitch_modification,
        "dct": _apply_dct_modification,
        "delay": _apply_delay_modification,
        "watermark": _apply_watermark_modification,
        "temporal": _apply_temporal_modification,
        "noise": _apply_noise_modification,
        "color": _apply_color_modification,
        "metadata": _apply_metadata_modification
    }
    
    if hash_type not in hash_functions:
        raise ValueError(f"Unknown hash type: {hash_type}. Available types: {list(hash_functions.keys())}")
        
    # Apply the selected modification function
    modified_clip = hash_functions[hash_type](clip, intensity)
    
    # Log the result
    logger.info(f"=== Completed {hash_type} hash modification ===")
    logger.info(f"Output clip properties: {modified_clip.size}, {modified_clip.duration}s, {modified_clip.fps}fps")
    
    return modified_clip

@timed
@log_exceptions
def _apply_pixel_modification(clip: VideoFileClip, intensity: float) -> VideoFileClip:
    """Apply pixel-level modification"""
    def pixel_modifier(frame):
        h, w = frame.shape[:2]
        
        # Create more interesting pattern with diagonal lines
        pattern_type = random.choice(['dots', 'diagonal', 'grid'])
        
        if pattern_type == 'dots':
            # Enhanced dot pattern
            step = max(1, int(12 / (intensity + 0.1)))  # Adjust step size based on intensity
            dot_size = max(1, int(2 * intensity))  # Size of dots scales with intensity
            
            for y in range(0, h, step):
                for x in range(0, w, step):
                    if (x + y) % 4 == 0:  # More sparse pattern
                        # Modify a block of pixels
                        y_end = min(y + dot_size, h)
                        x_end = min(x + dot_size, w)
                        # Use different color channels for visual interest
                        channel = random.choice([0, 1, 2])
                        change = int(3 + (intensity * 10))
                        frame[y:y_end, x:x_end, channel] = np.clip(
                            frame[y:y_end, x:x_end, channel] + change, 0, 255
                        )
        
        elif pattern_type == 'diagonal':
            # Diagonal line pattern
            thickness = max(1, int(intensity * 3))
            spacing = max(10, int(30 / (intensity + 0.2)))
            
            for offset in range(0, h + w, spacing):
                # Draw diagonal lines with varying channels
                channel = random.choice([0, 1, 2])
                change = int(5 + (intensity * 15))
                
                for t in range(thickness):
                    for i in range(h):
                        j = offset - i + t
                        if 0 <= j < w:
                            frame[i, j, channel] = np.clip(frame[i, j, channel] + change, 0, 255)
        
        else:  # grid
            # Grid pattern
            grid_size = max(15, int(40 / (intensity + 0.2)))
            line_width = max(1, int(intensity * 2))
            change = int(4 + (intensity * 12))
            
            # Draw horizontal and vertical grid lines
            for y in range(0, h, grid_size):
                y_range = range(max(0, y-line_width//2), min(h, y+line_width//2+1))
                channel = random.choice([0, 1, 2])
                for yr in y_range:
                    frame[yr, :, channel] = np.clip(frame[yr, :, channel] + change, 0, 255)
                    
            for x in range(0, w, grid_size):
                x_range = range(max(0, x-line_width//2), min(w, x+line_width//2+1))
                channel = random.choice([0, 1, 2])
                for xr in x_range:
                    frame[:, xr, channel] = np.clip(frame[:, xr, channel] + change, 0, 255)
        
        return frame
    
    return clip.fl_image(lambda f: process_frame_safely(f, pixel_modifier))

@timed
@log_exceptions
def _apply_glitch_modification(clip: VideoFileClip, intensity: float) -> VideoFileClip:
    """Apply glitch effect to the video"""
    def glitch_modifier(frame):
        h, w = frame.shape[:2]
        logger.debug(f"Processing frame of size {h}x{w}")
        
        # Validate frame type
        if frame.dtype != np.uint8:
            logger.warning(f"Frame dtype is {frame.dtype}, converting to uint8")
            frame = frame.astype(np.uint8)
        
        # Calculate glitch width based on intensity (increased from 0.02 to 0.05)
        glitch_width = max(5, int(w * intensity * 0.05))
        logger.debug(f"Glitch width: {glitch_width} pixels")
        
        if glitch_width > 0:
            # Randomly choose which side to apply the glitch
            side = random.choice(['left', 'right'])
            logger.debug(f"Applying glitch to {side} side")
            
            if side == 'left':
                glitch_start = random.randint(0, w//4)
                glitch_end = min(glitch_start + glitch_width, w)
                logger.debug(f"Left glitch area: {glitch_start} to {glitch_end}")
                
                shift_amount = random.randint(-10, 10)
                logger.debug(f"Vertical shift: {shift_amount} pixels")
                
                frame[:, glitch_start:glitch_end] = np.roll(
                    frame[:, glitch_start:glitch_end], 
                    shift_amount, 
                    axis=0
                )
                
                # Add color distortion
                frame[:, glitch_start:glitch_end, 0] = np.roll(
                    frame[:, glitch_start:glitch_end, 0], 
                    shift_amount + 2, 
                    axis=0
                )
            else:
                glitch_start = random.randint(3*w//4, w - glitch_width)
                glitch_end = min(glitch_start + glitch_width, w)
                logger.debug(f"Right glitch area: {glitch_start} to {glitch_end}")
                
                shift_amount = random.randint(-10, 10)
                logger.debug(f"Vertical shift: {shift_amount} pixels")
                
                frame[:, glitch_start:glitch_end] = np.roll(
                    frame[:, glitch_start:glitch_end], 
                    shift_amount, 
                    axis=0
                )
                
                frame[:, glitch_start:glitch_end, 2] = np.roll(
                    frame[:, glitch_start:glitch_end, 2], 
                    shift_amount - 2, 
                    axis=0
                )
        
        return frame
    
    return clip.fl_image(lambda f: process_frame_safely(f, glitch_modifier))

@timed
@log_exceptions
def _apply_dct_modification(clip: VideoFileClip, intensity: float) -> VideoFileClip:
    """Apply DCT-based modification"""
    def dct_modifier(frame):
        logger.debug(f"Processing frame of size {frame.shape[:2]}")
        
        # Validate frame type
        if frame.dtype != np.uint8:
            logger.warning(f"Frame dtype is {frame.dtype}, converting to uint8")
            frame = frame.astype(np.uint8)
        
        # Process all color channels
        modified_frame = frame.astype(float)
        
        for channel in range(3):
            logger.debug(f"Processing channel {channel}")
            
            # Apply DCT to the channel
            channel_data = modified_frame[:,:,channel]
            coeffs = dct(dct(channel_data.T, norm='ortho').T, norm='ortho')
            
            # Modify coefficients
            coeff_range = int(2 + intensity * 2)
            noise_scale = intensity * 0.5
            logger.debug(f"Modifying {coeff_range}x{coeff_range} coefficients with scale {noise_scale}")
            
            # Add controlled noise to the coefficients
            coeffs[0:coeff_range, 0:coeff_range] += np.random.normal(0, noise_scale, (coeff_range, coeff_range))
            
            # Apply inverse DCT
            modified_frame[:,:,channel] = idct(idct(coeffs.T, norm='ortho').T, norm='ortho')
        
        # Clip and convert back to uint8
        return np.clip(modified_frame, 0, 255).astype(np.uint8)
    
    return clip.fl_image(lambda f: process_frame_safely(f, dct_modifier))

@timed
@log_exceptions
def _apply_delay_modification(clip: VideoFileClip, intensity: float) -> VideoFileClip:
    """Apply delay modification"""
    # Increased delay amount
    delay_ms = int(10 + (intensity * 50))
    logger.debug(f"Adding delay of {delay_ms}ms")
    return clip.set_start(delay_ms/1000.0)

@timed
@log_exceptions
def _apply_watermark_modification(clip: VideoFileClip, intensity: float) -> VideoFileClip:
    """Apply watermark modification"""
    def watermark_modifier(frame):
        h, w = frame.shape[:2]
        logger.debug(f"Processing frame of size {h}x{w}")
        
        # Validate frame type
        if frame.dtype != np.uint8:
            logger.warning(f"Frame dtype is {frame.dtype}, converting to uint8")
            frame = frame.astype(np.uint8)
        
        # Calculate watermark size
        wm_size = min(32, h//8, w//8)
        logger.debug(f"Watermark size: {wm_size}x{wm_size}")
        
        # Create watermark pattern
        wm = np.zeros((wm_size, wm_size, 3), dtype=np.float32)
        center = wm_size // 2
        radius = wm_size // 4
        
        # Draw watermark pattern
        y, x = np.ogrid[:wm_size, :wm_size]
        mask = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2
        wm[mask] = [intensity * 0.8, intensity * 0.7, intensity * 0.9]
        
        # Choose position
        if random.random() < 0.7:
            corner = random.choice(['top-left', 'top-right', 'bottom-left', 'bottom-right'])
            if corner == 'top-left':
                pos_x, pos_y = 10, 10
            elif corner == 'top-right':
                pos_x, pos_y = w - wm_size - 10, 10
            elif corner == 'bottom-left':
                pos_x, pos_y = 10, h - wm_size - 10
            else:
                pos_x, pos_y = w - wm_size - 10, h - wm_size - 10
        else:
            pos_x = random.randint(0, w - wm_size)
            pos_y = random.randint(0, h - wm_size)
        
        logger.debug(f"Watermark position: ({pos_x}, {pos_y})")
        
        # Apply watermark
        overlay = frame[pos_y:pos_y+wm_size, pos_x:pos_x+wm_size].copy().astype(np.float32)
        overlay_strength = 35 + (intensity * 15)
        overlay = np.clip(overlay + (wm * overlay_strength), 0, 255)
        frame[pos_y:pos_y+wm_size, pos_x:pos_x+wm_size] = overlay.astype(np.uint8)
        
        return frame
    
    return clip.fl_image(lambda f: process_frame_safely(f, watermark_modifier))

@timed
@log_exceptions
def _apply_temporal_modification(clip: VideoFileClip, intensity: float) -> VideoFileClip:
    """Apply temporal modification"""
    def temporal_modifier(frame, t):
        if t % 1.0 < 0.1:
            logger.debug(f"Applying temporal modification at time {t}")
            # Increased intensity effect
            return frame * (1 + intensity * 0.5)
        return frame
    
    return clip.fl_image(lambda f: process_frame_safely(f, lambda x: temporal_modifier(x, time.time())))

@timed
@log_exceptions
def _apply_noise_modification(clip: VideoFileClip, intensity: float) -> VideoFileClip:
    """Apply noise modification"""
    def noise_modifier(frame):
        h, w = frame.shape[:2]
        logger.debug(f"Processing frame of size {h}x{w}")
        
        # Convert frame to float32 for processing
        frame_float = frame.astype(np.float32) / 255.0
        
        # Choose a noise pattern type
        noise_type = random.choice(['uniform', 'gaussian'])
        logger.debug(f"Selected noise type: {noise_type}")
        
        if noise_type == 'uniform':
            # Uniform noise across the entire frame with increased intensity
            noise = np.random.uniform(-0.1, 0.1, (h, w, 3)) * intensity
            logger.debug(f"Uniform noise range: {noise.min():.2f} to {noise.max():.2f}")
            frame_float = np.clip(frame_float + noise, 0, 1)
            
        else:  # gaussian
            # Gaussian noise with more noticeable focal points
            noise = np.random.normal(0, intensity * 0.1, (h, w, 3))
            logger.debug(f"Gaussian noise std: {noise.std():.2f}")
            
            # Create a single focal point where noise is stronger
            center_y = random.randint(0, h-1)
            center_x = random.randint(0, w-1)
            radius = random.randint(h//4, h//2)
            logger.debug(f"Focal point at ({center_x}, {center_y}) with radius {radius}")
            
            # Apply noise at focal point with falloff
            y_grid, x_grid = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((y_grid - center_y)**2 + (x_grid - center_x)**2)
            falloff = np.clip(1.0 - (dist_from_center / radius), 0, 1)
            falloff = falloff[:, :, np.newaxis]
            
            focal_noise = np.random.normal(0, intensity * 0.3, (h, w, 3)) * falloff
            logger.debug(f"Focal noise range: {focal_noise.min():.2f} to {focal_noise.max():.2f}")
            noise += focal_noise
            
            frame_float = np.clip(frame_float + noise, 0, 1)
        
        # Convert back to uint8
        return (frame_float * 255).astype(np.uint8)
    
    return clip.fl_image(lambda f: process_frame_safely(f, noise_modifier))

@timed
@log_exceptions
def _apply_color_modification(clip: VideoFileClip, intensity: float) -> VideoFileClip:
    """Apply color modification"""
    def color_modifier(frame):
        h, w = frame.shape[:2]
        logger.debug(f"Processing frame of size {h}x{w}")
        
        # Validate frame type
        if frame.dtype != np.uint8:
            logger.warning(f"Frame dtype is {frame.dtype}, converting to uint8")
            frame = frame.astype(np.uint8)
        
        # Choose color modification type
        color_type = random.choice(['pattern', 'shift', 'gradient', 'selective'])
        logger.debug(f"Selected color modification type: {color_type}")
        
        if color_type == 'pattern':
            pattern_style = random.choice(['grid', 'waves', 'blocks'])
            logger.debug(f"Selected pattern style: {pattern_style}")
            
            if pattern_style == 'grid':
                grid_size = max(8, int(30 / (intensity + 0.1)))
                logger.debug(f"Grid size: {grid_size}")
                
                pattern = np.zeros((h, w, 3), dtype=np.int8)
                for i in range(0, h, grid_size):
                    for j in range(0, w, grid_size):
                        if (i // grid_size + j // grid_size) % 2 == 0:
                            channel = random.randint(0, 2)
                            value = random.randint(2, int(5 + intensity * 15))
                            pattern[i:i+grid_size, j:j+grid_size, channel] = value
                
                frame = np.clip(frame + pattern, 0, 255)
            
            elif pattern_style == 'waves':
                for c in range(3):
                    frequency = random.uniform(1, 5) / w
                    phase = random.uniform(0, 2 * np.pi)
                    amplitude = random.uniform(1, intensity * 20)
                    logger.debug(f"Wave parameters for channel {c}: freq={frequency:.2f}, phase={phase:.2f}, amp={amplitude:.2f}")
                    
                    for j in range(w):
                        wave_val = amplitude * np.sin(2 * np.pi * frequency * j + phase)
                        wave_val = int(wave_val)
                        frame[:, j, c] = np.clip(frame[:, j, c] + wave_val, 0, 255)
            
            else:  # blocks
                num_blocks = random.randint(3, 10)
                logger.debug(f"Creating {num_blocks} color blocks")
                
                for _ in range(num_blocks):
                    block_h = random.randint(h//10, h//3)
                    block_w = random.randint(w//10, w//3)
                    start_y = random.randint(0, h - block_h)
                    start_x = random.randint(0, w - block_w)
                    
                    channel = random.randint(0, 2)
                    value = random.randint(2, int(5 + intensity * 10))
                    frame[start_y:start_y+block_h, start_x:start_x+block_w, channel] = \
                        np.clip(frame[start_y:start_y+block_h, start_x:start_x+block_w, channel] + value, 0, 255)
        
        elif color_type == 'shift':
            channel_shifts = []
            for _ in range(3):
                shift = random.uniform(-intensity * 8, intensity * 8)
                channel_shifts.append(shift)
            
            avg_shift = sum(channel_shifts) / 3
            channel_shifts = [s - avg_shift for s in channel_shifts]
            logger.debug(f"Channel shifts: {channel_shifts}")
            
            for c in range(3):
                frame[:, :, c] = np.clip(frame[:, :, c] + channel_shifts[c], 0, 255)
        
        elif color_type == 'gradient':
            direction = random.choice(['horizontal', 'vertical', 'radial'])
            logger.debug(f"Applying {direction} gradient")
            
            if direction == 'horizontal':
                for j in range(w):
                    factor = j / w
                    for c in range(3):
                        channel_effect = (random.uniform(-1, 1) * intensity * 10) * factor
                        frame[:, j, c] = np.clip(frame[:, j, c] + channel_effect, 0, 255)
            
            elif direction == 'vertical':
                for i in range(h):
                    factor = i / h
                    for c in range(3):
                        channel_effect = (random.uniform(-1, 1) * intensity * 10) * factor
                        frame[i, :, c] = np.clip(frame[i, :, c] + channel_effect, 0, 255)
            
            else:  # radial
                center_y, center_x = h // 2, w // 2
                max_dist = np.sqrt(center_y**2 + center_x**2)
                
                for i in range(h):
                    for j in range(w):
                        dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
                        factor = dist / max_dist
                        
                        for c in range(3):
                            channel_effect = (random.uniform(-1, 1) * intensity * 10) * factor
                            frame[i, j, c] = np.clip(frame[i, j, c] + channel_effect, 0, 255)
        
        else:  # selective color
            target_color = random.choice(['red', 'green', 'blue', 'yellow', 'cyan', 'magenta'])
            logger.debug(f"Selective color modification for {target_color}")
            
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            
            if target_color == 'red':
                mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
                mask2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
                mask = cv2.bitwise_or(mask1, mask2)
            elif target_color == 'green':
                mask = cv2.inRange(hsv, np.array([40, 50, 50]), np.array([80, 255, 255]))
            elif target_color == 'blue':
                mask = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([140, 255, 255]))
            elif target_color == 'yellow':
                mask = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([30, 255, 255]))
            elif target_color == 'cyan':
                mask = cv2.inRange(hsv, np.array([85, 50, 50]), np.array([95, 255, 255]))
            else:  # magenta
                mask = cv2.inRange(hsv, np.array([140, 50, 50]), np.array([160, 255, 255]))
            
            mask_3d = np.stack([mask, mask, mask], axis=2) / 255.0
            
            adjustment = np.zeros_like(frame, dtype=np.float32)
            adjustment[:, :, 0] = random.uniform(-intensity * 15, intensity * 15)
            adjustment[:, :, 1] = random.uniform(-intensity * 15, intensity * 15)
            adjustment[:, :, 2] = random.uniform(-intensity * 15, intensity * 15)
            
            frame = np.clip(frame + adjustment * mask_3d, 0, 255).astype(np.uint8)
        
        return frame
    
    return clip.fl_image(lambda f: process_frame_safely(f, color_modifier))

@timed
@log_exceptions
def _apply_metadata_modification(clip: VideoFileClip, intensity: float) -> VideoFileClip:
    """Apply metadata modification"""
    logger.info(f"Original metadata: fps={clip.fps}, size={clip.size}, duration={clip.duration}")
    
    # More noticeable FPS modification
    fps_mod = clip.fps + (random.uniform(-0.5, 0.5) * intensity)
    logger.debug(f"FPS modification: {clip.fps} -> {fps_mod}")
    clip = clip.set_fps(fps_mod)
    
    # More noticeable resolution modification
    scale = 1 + (random.uniform(-0.05, 0.05) * intensity)
    new_w = int(clip.w * scale)
    new_h = int(clip.h * scale)
    logger.debug(f"Resolution modification: {clip.size} -> ({new_w}, {new_h})")
    clip = clip.resize(width=new_w, height=new_h)
    
    # Modify video properties
    duration_mod = clip.duration + (random.uniform(-0.01, 0.01) * intensity)
    logger.debug(f"Duration modification: {clip.duration} -> {duration_mod}")
    clip = clip.set_duration(duration_mod)
    
    # Add random rotation
    rotation = random.uniform(-0.5, 0.5) * intensity
    if abs(rotation) > 0.01:
        logger.debug(f"Applying rotation: {rotation} degrees")
        clip = clip.rotate(rotation)
    
    logger.info(f"Modified metadata: fps={clip.fps}, size={clip.size}, duration={clip.duration}")
    return clip 