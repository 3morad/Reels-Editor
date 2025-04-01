import os
from typing import Dict, Tuple, Optional
from moviepy.editor import VideoFileClip
import cv2
import numpy as np

class VideoInput:
    def __init__(self, input_path: str):
        self.input_path = input_path
        self.video_clip = None
        self.properties = {}
        self.validate()  # Initialize video clip immediately

    def validate(self) -> bool:
        """Validate if the input file exists and is a valid video file."""
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Video file not found: {self.input_path}")
        
        try:
            if self.video_clip is None:
                self.video_clip = VideoFileClip(self.input_path)
                
            # Verify video clip has required attributes
            if not hasattr(self.video_clip, 'h') or not hasattr(self.video_clip, 'w'):
                raise ValueError("Video clip missing required attributes")
                
            # Verify video clip can get frames
            test_frame = self.video_clip.get_frame(0)
            if not isinstance(test_frame, np.ndarray):
                raise ValueError("Video clip cannot get valid frames")
                
            return True
        except Exception as e:
            if self.video_clip:
                self.video_clip.close()
            self.video_clip = None
            raise ValueError(f"Invalid video file: {str(e)}")

    def analyze(self) -> Dict:
        """Analyze video properties including duration, resolution, and fps."""
        if not self.video_clip:
            self.validate()

        try:
            self.properties = {
                "duration": self.video_clip.duration,
                "width": self.video_clip.w,
                "height": self.video_clip.h,
                "fps": self.video_clip.fps,
                "size": self.video_clip.size,
                "rotation": self.video_clip.rotation
            }
            return self.properties
        except Exception as e:
            raise ValueError(f"Error analyzing video properties: {str(e)}")

    def get_frame_at_time(self, time: float) -> Optional[np.ndarray]:
        """Get a specific frame from the video at the given time."""
        if not self.video_clip:
            self.validate()
        
        try:
            frame = self.video_clip.get_frame(time)
            if not isinstance(frame, np.ndarray):
                raise ValueError("Invalid frame type")
            return frame
        except Exception as e:
            print(f"Error getting frame at time {time}: {str(e)}")
            return None

    def close(self):
        """Close the video clip to free up resources."""
        if self.video_clip:
            self.video_clip.close()
            self.video_clip = None

    def __enter__(self):
        self.validate()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close() 