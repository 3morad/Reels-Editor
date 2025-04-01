from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
import numpy as np
from TTS.api import TTS
import soundfile as sf
import tempfile
import os

@dataclass
class TextSegment:
    text: str
    timestamp: float
    duration: float
    confidence: float

class VoiceOverGenerator:
    def __init__(self, model_name: str = "tts_models/en/ljspeech/tacotron2-DDC"):
        """Initialize the voice-over generator with specified TTS model."""
        self.tts = TTS(model_name=model_name)
        self.temp_dir = tempfile.mkdtemp()
        
    def extract_text_segments(self, video_clip: VideoFileClip) -> List[TextSegment]:
        """Extract text and timing from video frames."""
        segments = []
        fps = video_clip.fps
        
        def process_frame(frame, t):
            print(f"\nProcessing voice-over frame at time {t}")
            # Use existing text detection
            detected_text = self.text_processor.detect_text_regions(frame)
            if detected_text:
                segments.append(TextSegment(
                    text=" ".join([r.text for r in detected_text]),
                    timestamp=t,
                    duration=1/fps,
                    confidence=min([r.confidence for r in detected_text])
                ))
            return frame
            
        # Use fl instead of fl_image to get time parameter
        video_clip.fl(process_frame)
        return self._merge_continuous_segments(segments)
    
    def _merge_continuous_segments(self, segments: List[TextSegment], 
                                 time_threshold: float = 0.5) -> List[TextSegment]:
        """Merge text segments that appear close in time."""
        if not segments:
            return []
            
        merged = []
        current = segments[0]
        
        for next_seg in segments[1:]:
            if next_seg.timestamp - (current.timestamp + current.duration) < time_threshold:
                current.text += " " + next_seg.text
                current.duration = next_seg.timestamp + next_seg.duration - current.timestamp
                current.confidence = max(current.confidence, next_seg.confidence)
            else:
                merged.append(current)
                current = next_seg
                
        merged.append(current)
        return merged
    
    def generate_voice_over(self, text_segments: List[TextSegment], 
                          speed: float = 1.0,
                          pitch: float = 1.0) -> AudioFileClip:
        """Generate voice-over audio for text segments."""
        audio_segments = []
        
        for segment in text_segments:
            if segment.confidence < 0.6:  # Skip low confidence detections
                continue
                
            # Generate speech for each segment
            temp_path = os.path.join(self.temp_dir, f"segment_{len(audio_segments)}.wav")
            self.tts.tts_to_file(text=segment.text, 
                               file_path=temp_path,
                               speed=speed)
            
            # Load the generated audio
            audio = AudioFileClip(temp_path)
            
            # Adjust pitch if needed
            if pitch != 1.0:
                audio = audio.set_pitch(pitch)
            
            # Set timing
            audio = audio.set_start(segment.timestamp)
            
            audio_segments.append(audio)
        
        # Combine all segments
        if not audio_segments:
            return None
            
        final_audio = CompositeAudioClip(audio_segments)
        return final_audio
    
    def add_to_video(self, video_clip: VideoFileClip,
                    speed: float = 1.0,
                    pitch: float = 1.0,
                    original_audio_volume: float = 0.3) -> VideoFileClip:
        """Add AI voice-over to video with specified parameters."""
        # Extract text segments
        text_segments = self.extract_text_segments(video_clip)
        
        # Generate voice-over
        voice_over = self.generate_voice_over(text_segments, speed, pitch)
        
        if voice_over is None:
            return video_clip
            
        # Mix with original audio if exists
        if video_clip.audio:
            original_audio = video_clip.audio.volumex(original_audio_volume)
            final_audio = CompositeAudioClip([original_audio, voice_over])
        else:
            final_audio = voice_over
            
        # Apply to video
        return video_clip.set_audio(final_audio)
    
    def cleanup(self):
        """Clean up temporary files."""
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir) 