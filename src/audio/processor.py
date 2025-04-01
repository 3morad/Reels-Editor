from typing import Optional, Tuple
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
from pydub import AudioSegment
import numpy as np

class AudioProcessor:
    def __init__(self, video_clip: VideoFileClip):
        self.video_clip = video_clip
        self.audio_clip = video_clip.audio
        self.modified_clip = None
        self.modifications = []

    def adjust_volume(self, volume_factor: float) -> 'AudioProcessor':
        """Adjust the volume of the audio."""
        if self.audio_clip is None:
            raise ValueError("No audio track found in video")

        self.modified_clip = self.audio_clip.volumex(volume_factor)
        self.modifications.append({
            'type': 'volume_adjustment',
            'factor': volume_factor
        })
        return self

    def add_background_music(self, music_path: str, volume_factor: float = 0.3,
                           loop: bool = True) -> 'AudioProcessor':
        """Add background music to the video."""
        if self.audio_clip is None:
            raise ValueError("No audio track found in video")

        background = AudioFileClip(music_path)
        
        if loop:
            # Loop the background music if it's shorter than the video
            if background.duration < self.video_clip.duration:
                n_loops = int(np.ceil(self.video_clip.duration / background.duration))
                background = background.loop(n=n_loops)
            # Trim to match video duration
            background = background.subclip(0, self.video_clip.duration)
        
        # Adjust background music volume
        background = background.volumex(volume_factor)
        
        # Combine original audio with background music
        self.modified_clip = CompositeAudioClip([self.audio_clip, background])
        self.modifications.append({
            'type': 'background_music',
            'music_path': music_path,
            'volume_factor': volume_factor,
            'loop': loop
        })
        return self

    def add_audio_effect(self, effect_type: str, **kwargs) -> 'AudioProcessor':
        """Add audio effects like echo, reverb, etc."""
        if self.audio_clip is None:
            raise ValueError("No audio track found in video")

        if effect_type == 'echo':
            delay = kwargs.get('delay', 0.1)
            decay = kwargs.get('decay', 0.5)
            # Convert to numpy array for processing
            audio_array = self.audio_clip.to_soundarray()
            delayed = np.zeros_like(audio_array)
            delayed[int(delay * self.audio_clip.fps):] = audio_array[:-int(delay * self.audio_clip.fps)] * decay
            self.modified_clip = self.audio_clip.set_array(audio_array + delayed)
        else:
            raise ValueError(f"Unsupported audio effect: {effect_type}")

        self.modifications.append({
            'type': 'audio_effect',
            'effect_type': effect_type,
            'params': kwargs
        })
        return self

    def fade_in(self, duration: float = 1.0) -> 'AudioProcessor':
        """Add fade in effect to the audio."""
        if self.audio_clip is None:
            raise ValueError("No audio track found in video")

        self.modified_clip = self.audio_clip.fadein(duration)
        self.modifications.append({
            'type': 'fade_in',
            'duration': duration
        })
        return self

    def fade_out(self, duration: float = 1.0) -> 'AudioProcessor':
        """Add fade out effect to the audio."""
        if self.audio_clip is None:
            raise ValueError("No audio track found in video")

        self.modified_clip = self.audio_clip.fadeout(duration)
        self.modifications.append({
            'type': 'fade_out',
            'duration': duration
        })
        return self

    def get_modified_clip(self) -> Optional[AudioFileClip]:
        """Get the modified audio clip."""
        return self.modified_clip if self.modified_clip is not None else self.audio_clip

    def get_modifications(self) -> list:
        """Get the list of applied audio modifications."""
        return self.modifications

    def reset(self) -> 'AudioProcessor':
        """Reset all audio modifications."""
        self.modified_clip = None
        self.modifications = []
        return self 