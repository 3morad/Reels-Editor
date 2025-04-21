from typing import Dict, List, Set
from ..utils.logging_utils import configure_logger

logger = configure_logger("HashPresets")

# Define the presets and their methods
HASH_PRESETS = {
    "fast": {
        "name": "Fast Processing",
        "description": "Quick processing with basic hash modifications",
        # Only watermark, glitch, and noise for fastest mode
        "methods": ["watermark", "glitch", "noise"],
        "default_intensities": {
            "watermark": 0.4,
            "glitch": 0.2,
            "noise": 0.2
        }
    },
    "normal": {
        "name": "Balanced Processing",
        "description": "Balanced approach between speed and effectiveness",
        "methods": ["metadata", "delay", "glitch", "pixelate", "noise"],
        "default_intensities": {
            "metadata": 0.8,
            "delay": 0.6,
            "glitch": 0.2,
            "pixelate": 0.3,
            "noise": 0.2
        }
    },
    "slow": {
        "name": "Maximum Effectiveness",
        "description": "Uses all methods for maximum hash modification",
        "methods": ["metadata", "delay", "glitch", "pixelate", "noise", "watermark", "color", "temporal", "dct"],
        "default_intensities": {
            "metadata": 0.8,
            "delay": 0.6,
            "glitch": 0.2,
            "pixelate": 0.3,
            "noise": 0.3,
            "watermark": 0.4,
            "color": 0.2,
            "temporal": 0.4,
            "dct": 0.3
        }
    }
}


def get_preset_methods(preset: str) -> List[str]:
    """Get the list of hash methods for a given preset."""
    if preset not in HASH_PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Available presets: {list(HASH_PRESETS.keys())}")
    return HASH_PRESETS[preset]["methods"]

def get_preset_default_intensity(preset: str, method: str) -> float:
    """Get the default intensity for a method in a given preset."""
    if preset not in HASH_PRESETS:
        raise ValueError(f"Unknown preset: {preset}")
    if method not in HASH_PRESETS[preset]["default_intensities"]:
        raise ValueError(f"Method {method} not found in preset {preset}")
    return HASH_PRESETS[preset]["default_intensities"][method]

def get_preset_info(preset: str) -> Dict:
    """Get information about a preset."""
    if preset not in HASH_PRESETS:
        raise ValueError(f"Unknown preset: {preset}")
    return HASH_PRESETS[preset] 