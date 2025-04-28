"""
FFmpeg hash and distortion effects for video processing.
These effects create various types of visual distortions and hash-like effects.
"""

from typing import List, Tuple, Optional
import logging
import random

logger = logging.getLogger(__name__)

def get_pixelate_params(intensity: float = 0.5) -> List[str]:
    """
    Creates a subtle pixelation effect with random areas and patterns.
    Args:
        intensity: Effect intensity (0.1 to 1.0)
    """
    # Increase intensity slightly for better hash differences
    subtle_intensity = intensity * 0.7  # Increase from 0.5 to 0.7
    
    # Choose between different pixelation techniques at random
    technique = random.randint(0, 2)
    
    if technique == 0:
        # Standard pixelation with randomized size
        pixel_size = int(4 + subtle_intensity * random.uniform(4, 8))  # Slightly larger pixels
        return [
            '-vf', f'scale=iw/{pixel_size}:ih/{pixel_size},scale=iw:ih:flags=neighbor'
        ]
    elif technique == 1:
        # Partial pixelation - affects slightly larger part of frame
        x_pos = random.randint(10, 90)  # % of frame width
        y_pos = random.randint(10, 90)  # % of frame height 
        width = random.randint(8, 20)   # Increased from 5-15 to 8-20
        height = random.randint(8, 20)  # Increased from 5-15 to 8-20
        pixel_size = int(3 + subtle_intensity * random.uniform(3, 6))
        
        # Create a complex filter chain for partial pixelation
        return ['-vf', f'split [a][b]; [a]crop=iw*{width/100}:ih*{height/100}:iw*{x_pos/100}:ih*{y_pos/100},scale=iw/{pixel_size}:ih/{pixel_size},scale=iw:ih:flags=neighbor[pixelated]; [b][pixelated]overlay=x=iw*{x_pos/100}:y=ih*{y_pos/100}']
    else:
        # Mosaic pattern with slightly larger blocks
        blocks = random.randint(3, 10)  # Increased from 2-8 to 3-10
        return [
            '-vf', f'format=rgb24,boxblur={blocks}:1:cr=0:ar=0:lr=0,format=yuv420p'
        ]

def get_glitch_params(intensity: float = 0.5) -> list:
    """
    Create a dynamic glitching effect with vertical lines on the left and right sides of the screen.
    Uses simpler FFmpeg expressions for better compatibility.
    Args:
        intensity: Effect intensity (0.1 to 1.0)
    """
    # Increase intensity slightly
    subtle_intensity = intensity * 0.6
    
    # Only use vertical lines option now
    num_lines = random.randint(3, max(4, int(5 * subtle_intensity)))
    lines = []
    
    # Generate lines for left side
    for i in range(num_lines):
        # Random vertical position on left side
        x_pos = random.randint(10, 150)
        
        # Line thickness
        thickness = random.randint(1, 3)
        
        # Semi-transparent lines
        opacity = random.uniform(0.1, 0.2)
        
        # Random color for each line
        r = random.randint(180, 255)
        g = random.randint(180, 255)
        b = random.randint(180, 255)
        color = f"0x{r:02x}{g:02x}{b:02x}@{opacity}"
        
        # Create vertical line with timing
        mod_value = random.randint(4, 10)  # Controls frequency
        lines.append(f'drawbox=x={x_pos}:y=0:w={thickness}:h=ih:color={color}:t=fill:enable=\'not(mod(floor(t*24),{mod_value}))\'')
    
    # Generate lines for right side
    for i in range(num_lines):
        # Random vertical position on right side (using relative positioning with iw)
        x_pos_rel = random.uniform(0.7, 0.95)
        
        # Line thickness
        thickness = random.randint(1, 3)
        
        # Semi-transparent lines
        opacity = random.uniform(0.1, 0.2)
        
        # Random color for each line
        r = random.randint(180, 255)
        g = random.randint(180, 255)
        b = random.randint(180, 255)
        color = f"0x{r:02x}{g:02x}{b:02x}@{opacity}"
        
        # Create vertical line with timing (using iw to position relative to right side)
        mod_value = random.randint(4, 10)
        offset = random.randint(0, 5)
        lines.append(f'drawbox=x=iw*{x_pos_rel}:y=0:w={thickness}:h=ih:color={color}:t=fill:enable=\'not(mod(floor(t*24+{offset}),{mod_value}))\'')
    
    return ['-vf', ','.join(lines)]

def get_noise_params(intensity: float = 0.5) -> List[str]:
    """
    Adds a subtle but visible grain overlay to the video.
    Args:
        intensity: Effect intensity (0.1 to 1.0)
    """
    # Use a moderate intensity for a subtle but visible grain
    adjusted_intensity = intensity * 0.8  # More moderate intensity
    
    # Lower noise values for a lighter grain effect
    strength = max(2, int(adjusted_intensity * random.uniform(2, 5)))  # Lower strength values
    
    # Create different noise patterns
    noise_type = random.randint(0, 2)
    
    if noise_type == 0:
        # Light uniform noise
        return ['-vf', f'noise=c0s={strength}:c1s={strength}:c2s={strength}:allf=a']
    elif noise_type == 1:
        # Subtle color noise variation
        r_noise = max(2, int(strength * random.uniform(0.9, 1.1)))
        g_noise = max(2, int(strength * random.uniform(0.9, 1.1)))
        b_noise = max(2, int(strength * random.uniform(0.9, 1.1)))
        return ['-vf', f'noise=c0s={r_noise}:c1s={g_noise}:c2s={b_noise}:allf=a']
    else:
        # Temporal noise with moderate settings
        temp_strength = max(2, strength - 1)  # Slightly lower for temporal
        return ['-vf', f'noise=c0s={temp_strength}:c1s={temp_strength}:c2s={temp_strength}:allf=t']

def get_color_params(intensity: float = 0.5) -> List[str]:
    """
    Creates very subtle color distortion effect with randomized parameters.
    Args:
        intensity: Effect intensity (0.1 to 1.0)
    """
    # Reduce intensity for subtlety
    subtle_intensity = intensity * 0.3  # Reduce intensity by 70%
    
    # Randomize all color parameters but keep very subtle
    brightness = round(random.uniform(-0.05, 0.05) * subtle_intensity, 3)
    contrast = round(random.uniform(0.97, 1.03), 3)
    saturation = round(random.uniform(0.97, 1.03), 3)
    
    # Randomly select additional color effect
    effect_type = random.randint(0, 2)
    
    if effect_type == 0:
        # Standard color adjustment
        return [
            '-vf', f'eq=brightness={brightness}:contrast={contrast}:saturation={saturation}'
        ]
    elif effect_type == 1:
        # Add very mild gamma adjustment
        gamma = round(random.uniform(0.97, 1.03), 3)
        return [
            '-vf', f'eq=brightness={brightness}:contrast={contrast}:saturation={saturation}:gamma={gamma}'
        ]
    else:
        # Add very mild hue adjustment
        hue = random.randint(0, 5)
        return [
            '-vf', f'eq=brightness={brightness}:contrast={contrast}:saturation={saturation},hue=h={hue}'
        ]

def get_watermark_params(intensity: float = 0.5) -> List[str]:
    """
    Adds watermarks in the four corners of the video that appear and disappear.
    Args:
        intensity: Effect intensity (0.1 to 1.0)
    """
    # Create watermarks for each corner
    watermarks = []
    
    # Define the four corners
    corners = [
        # Top-left corner
        {"x_min": 10, "x_max": 150, "y_min": 10, "y_max": 150},
        # Top-right corner (using relative positioning with iw)
        {"x_min_rel": 0.85, "x_max_rel": 0.97, "y_min": 10, "y_max": 150},
        # Bottom-left corner
        {"x_min": 10, "x_max": 150, "y_min_rel": 0.85, "y_max_rel": 0.97},
        # Bottom-right corner (using relative positioning with iw and ih)
        {"x_min_rel": 0.85, "x_max_rel": 0.97, "y_min_rel": 0.85, "y_max_rel": 0.97}
    ]
    
    # Create 1-2 watermarks for each corner
    for corner in corners:
        # Number of watermarks in this corner (1-2)
        num_corner_marks = random.randint(1, 2)
        
        for i in range(num_corner_marks):
            # Position in corner
            if "x_min_rel" in corner:
                # Use relative positioning (percentage of video width)
                x = f"iw*{random.uniform(corner['x_min_rel'], corner['x_max_rel']):.4f}"
            else:
                # Use absolute positioning (pixels)
                x = random.randint(corner["x_min"], corner["x_max"])
                
            if "y_min_rel" in corner:
                # Use relative positioning (percentage of video height)
                y = f"ih*{random.uniform(corner['y_min_rel'], corner['y_max_rel']):.4f}"
            else:
                # Use absolute positioning (pixels)
                y = random.randint(corner["y_min"], corner["y_max"])
            
            # Moderate opacity for visibility without being too distracting
            opacity = random.uniform(0.05, 0.08)
            
            # Moderate size
            w = random.randint(30, 70)
            h = random.randint(30, 70)
            
            # Bright, visible colors
            r = random.randint(200, 255)
            g = random.randint(200, 255)
            b = random.randint(200, 255)
            color = f"0x{r:02x}{g:02x}{b:02x}@{opacity}"
            
            # Timing pattern
            mod_value = random.randint(3, 10)  # Controls frequency
            offset = random.randint(0, 5)
            
            # Generate the timing expression
            timeline = f'not(mod(floor(t*24+{offset}),{mod_value}))'
            
            # Fill type
            fill_type = "fill" if random.random() < 0.7 else "1"
            
            # Add watermark with its timing
            watermarks.append(f'drawbox=x={x}:y={y}:w={w}:h={h}:color={color}:t={fill_type}:enable=\'{timeline}\'')
            
            # Maybe add an inner box for more visual interest
            if random.random() < 0.6:
                inner_w = max(5, w - random.randint(10, 20))
                inner_h = max(5, h - random.randint(10, 20))
                
                # Using expressions for centering the inner box
                if isinstance(x, str) and x.startswith("iw"):
                    # For relative positioning, use expressions
                    inner_x = f"({x})+({w-inner_w})/2"
                else:
                    # For absolute positioning, calculate
                    inner_x = x + (w - inner_w) // 2
                    
                if isinstance(y, str) and y.startswith("ih"):
                    # For relative positioning, use expressions
                    inner_y = f"({y})+({h-inner_h})/2"
                else:
                    # For absolute positioning, calculate
                    inner_y = y + (h - inner_h) // 2
                
                inner_color = f"0x{b:02x}{r:02x}{g:02x}@{opacity * 1.2}"
                watermarks.append(f'drawbox=x={inner_x}:y={inner_y}:w={inner_w}:h={inner_h}:color={inner_color}:t={fill_type}:enable=\'{timeline}\'')
    
    # Combine all watermarks
    return ['-vf', ','.join(watermarks)]

def get_metadata_params(intensity: float = 0.5) -> List[str]:
    """
    Simulates metadata corruption and adds random metadata.
    Args:
        intensity: Effect intensity (0.1 to 1.0)
    """
    # Video technical modifications - only FPS changes, no rotation
    fps = random.choice([15, 24, 29.97, 30, 60])
    filters = []
    filters.append(f'fps={fps}')
    
    # Random trim but no rotation
    if random.random() < 0.5:
        filters.append('trim=end=5')

    # Random metadata generation
    metadata_params = []
    
    # Random timestamp between 2024-2025
    year = random.randint(2024, 2025)
    month = random.randint(1, 12)
    day = random.randint(1, 28)  # Using 28 to be safe with February
    hour = random.randint(0, 23)
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    creation_time = f"{year}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}"
    metadata_params.extend(['-metadata', f'creation_time={creation_time}'])
    
    # Random location (coordinates within reasonable bounds)
    lat = random.uniform(-90, 90)
    lon = random.uniform(-180, 180)
    metadata_params.extend(['-metadata', f'location={lat:.6f}/{lon:.6f}'])
    
    # Random device model
    devices = [
        'iPhone 13 Pro', 'iPhone 14 Pro Max', 'iPhone 15', 'Samsung Galaxy S23', 
        'Google Pixel 7', 'Samsung Galaxy S24 Ultra', 'iPhone 12 Pro'
    ]
    device = random.choice(devices)
    metadata_params.extend(['-metadata', f'device_model={device}'])
    
    # Random comments
    comments = [
        'Original content', 'Created with love', 'My video creation', 
        'Edited version', 'Final cut', 'Draft version', 'Content creation',
        'Social media content', 'Original footage'
    ]
    comment = random.choice(comments)
    metadata_params.extend(['-metadata', f'comment={comment}'])
    
    # Random tags/artist
    artists = ['Smartphone', 'Mobile Camera', 'Digital Creator', 'Content Creator']
    artist = random.choice(artists)
    metadata_params.extend(['-metadata', f'artist={artist}'])
    
    # Random title (60% chance)
    if random.random() < 0.6:
        titles = ['Video', 'Clip', 'Recording', 'Footage', 'Content']
        title = random.choice(titles)
        metadata_params.extend(['-metadata', f'title={title}'])

    # Return video filters and metadata parameters separately
    # Only include '-vf' if we have filters
    result = []
    if filters:
        result.extend(['-vf', ','.join(filters)])
    result.extend(metadata_params)
    return result

def get_delay_params(intensity: float = 0.5) -> List[str]:
    """
    Creates frame delay effect.
    Args:
        intensity: Effect intensity (0.1 to 1.0)
    """
    delay_frames = int(10 + intensity * 30)
    return ['-vf', f'tpad=start_duration={delay_frames/30.0}:color=black']

def get_temporal_params(intensity: float = 0.5) -> List[str]:
    """
    Creates temporal distortion effect.
    Args:
        intensity: Effect intensity (0.1 to 1.0)
    """
    frames = int(2 + intensity * 8)  # 2-10 frames
    return [
        '-vf', f'tblend=all_mode=average,framestep={frames}'
    ]

def get_fallback_params() -> List[str]:
    """
    Generate a simple, guaranteed-to-work effect for when other effects fail.
    Returns a combination of very basic FFmpeg filters that should work on all versions.
    """
    effect = random.randint(0, 3)
    
    if effect == 0:
        # Slight brightness change
        brightness = random.uniform(-0.05, 0.05)
        return ['-vf', f'eq=brightness={brightness}:contrast=1.0']
    elif effect == 1:
        # Slight crop
        crop_percent = 0.03
        return ['-vf', f'crop=iw*{1-crop_percent*2}:ih*{1-crop_percent*2}']
    elif effect == 2:
        # Basic hue shift
        hue = random.randint(0, 30)
        return ['-vf', f'hue=h={hue}']
    else:
        # Minimal pixelation
        return ['-vf', 'scale=iw/4:ih/4,scale=iw:ih']

def get_hash_params(hash_type: str, intensity: float = 1.0) -> List[str]:
    """
    Get FFmpeg filter parameters for hash effects.
    Args:
        hash_type: Type of hash effect to apply
        intensity: Effect intensity (0.1 to 1.0)
    """
    intensity = max(0.1, min(1.0, intensity))  # Clamp intensity between 0.1 and 1.0
    
    try:
        if hash_type == 'pixelate':
            return get_pixelate_params(intensity)
        elif hash_type == 'glitch':
            return get_glitch_params(intensity)
        elif hash_type == 'noise':
            return get_noise_params(intensity)
        elif hash_type == 'color':
            return get_color_params(intensity)
        elif hash_type == 'watermark':
            return get_watermark_params(intensity)
        elif hash_type == 'metadata':
            return get_metadata_params(intensity)
        elif hash_type == 'delay':
            return get_delay_params(intensity)
        elif hash_type == 'temporal':
            return get_temporal_params(intensity)
        elif hash_type == 'fallback':
            return get_fallback_params()
        else:
            logger.warning(f"Unknown hash type: {hash_type}, using fallback")
            return get_fallback_params()
    except Exception as e:
        # If any effect fails, use the fallback
        logger.warning(f"Error applying effect {hash_type}: {e}, using fallback")
        return get_fallback_params() 