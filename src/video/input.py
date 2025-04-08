import os
import logging
import time
from typing import Dict, Tuple, Optional
from moviepy.editor import VideoFileClip
import cv2
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("VideoInput")

class VideoInput:
    def __init__(self, input_path: str):
        self.start_time = time.time()
        logger.debug(f"Initializing VideoInput with path: {input_path}")
        self.input_path = input_path
        self.video_clip = None
        self.properties = {}
        self.validate()  # Initialize video clip immediately
        logger.info(f"Initialization took {time.time() - self.start_time:.2f} seconds")

    def validate(self) -> bool:
        """Validate if the input file exists and is a valid video file."""
        validation_start = time.time()
        logger.debug(f"Validating video file: {self.input_path}")
        
        if not os.path.exists(self.input_path):
            logger.error(f"Video file not found: {self.input_path}")
            raise FileNotFoundError(f"Video file not found: {self.input_path}")
        
        try:
            if self.video_clip is None:
                logger.debug("Creating VideoFileClip object")
                clip_start = time.time()
                self.video_clip = VideoFileClip(self.input_path)
                logger.info(f"VideoFileClip creation took {time.time() - clip_start:.2f} seconds")
                
            # Verify video clip has required attributes
            logger.debug("Checking video clip attributes")
            if not hasattr(self.video_clip, 'h') or not hasattr(self.video_clip, 'w'):
                logger.error("Video clip missing required attributes (height/width)")
                raise ValueError("Video clip missing required attributes")
            
            # Verify video clip has duration
            if not hasattr(self.video_clip, 'duration') or self.video_clip.duration <= 0:
                logger.error(f"Invalid video duration: {getattr(self.video_clip, 'duration', 'None')}")
                raise ValueError("Video clip has invalid duration")
                
            # Verify video clip can get frames
            logger.debug("Testing frame extraction")
            frame_start = time.time()
            test_frame = self.video_clip.get_frame(0)
            logger.info(f"First frame extraction took {time.time() - frame_start:.2f} seconds")
            
            if not isinstance(test_frame, np.ndarray):
                logger.error(f"Invalid frame type: {type(test_frame)}")
                raise ValueError("Video clip cannot get valid frames")
                
            logger.info(f"Video validation successful. Dimensions: {self.video_clip.w}x{self.video_clip.h}, Duration: {self.video_clip.duration}s")
            logger.info(f"Total validation took {time.time() - validation_start:.2f} seconds")
            return True
        except Exception as e:
            logger.error(f"Video validation failed: {str(e)}", exc_info=True)
            if self.video_clip:
                self.video_clip.close()
            self.video_clip = None
            raise ValueError(f"Invalid video file: {str(e)}")

    def analyze(self) -> Dict:
        """Analyze video properties including duration, resolution, and fps."""
        analysis_start = time.time()
        logger.debug("Starting video analysis")
        if not self.video_clip:
            logger.debug("Video clip not initialized, running validation")
            self.validate()

        try:
            logger.debug("Extracting video properties")
            self.properties = {
                "duration": self.video_clip.duration,
                "width": self.video_clip.w,
                "height": self.video_clip.h,
                "fps": self.video_clip.fps,
                "size": self.video_clip.size,
                "rotation": self.video_clip.rotation
            }
            logger.info(f"Video analysis complete: {self.properties}")
            logger.info(f"Analysis took {time.time() - analysis_start:.2f} seconds")
            return self.properties
        except Exception as e:
            logger.error(f"Error analyzing video properties: {str(e)}", exc_info=True)
            raise ValueError(f"Error analyzing video properties: {str(e)}")

    def get_frame_at_time(self, time: float) -> Optional[np.ndarray]:
        """Get a specific frame from the video at the given time."""
        frame_start = time.time()
        logger.debug(f"Getting frame at time: {time}")
        if not self.video_clip:
            logger.debug("Video clip not initialized, running validation")
            self.validate()
        
        try:
            frame = self.video_clip.get_frame(time)
            if not isinstance(frame, np.ndarray):
                logger.error(f"Invalid frame type: {type(frame)}")
                raise ValueError("Invalid frame type")
            logger.debug(f"Successfully retrieved frame at time {time} in {time.time() - frame_start:.2f} seconds")
            return frame
        except Exception as e:
            logger.error(f"Error getting frame at time {time}: {str(e)}", exc_info=True)
            return None

    def close(self):
        """Close the video clip to free up resources."""
        logger.debug("Closing video clip")
        if self.video_clip:
            self.video_clip.close()
            self.video_clip = None
            logger.info(f"Video clip closed successfully. Total lifetime: {time.time() - self.start_time:.2f} seconds")

    def __enter__(self):
        logger.debug("Entering context manager")
        self.validate()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.debug("Exiting context manager")
        self.close() 