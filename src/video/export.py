import os
import json
from datetime import datetime
from typing import Dict, Optional, Any
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
import time
import logging
import multiprocessing
import shutil

# Configure logging
logger = logging.getLogger("VideoExporter")

class VideoExporter:
    def __init__(self, output_dir: str = 'output'):
        self.start_time = time.time()
        logger.info(f"Initializing VideoExporter with output directory: {output_dir}")
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        # Get CPU count for optimal thread usage
        self.cpu_count = max(1, multiprocessing.cpu_count() - 1)
        logger.info(f"Using {self.cpu_count} threads for video processing")
        logger.info(f"VideoExporter initialized in {time.time() - self.start_time:.2f}s")

    def generate_filename(self, original_filename: str, variation: int = 0) -> str:
        """Generate a unique filename for the exported video."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = os.path.splitext(original_filename)[0]
        return f"{base_name}_v{variation}_{timestamp}.mp4"

    def save_metadata(self, metadata: Dict, filename: str):
        """Save metadata to a JSON file."""
        metadata_path = os.path.join(self.output_dir, f"{os.path.splitext(filename)[0]}_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)

    def export_video(self, 
                    video_clip, 
                    audio_clip: AudioFileClip = None, 
                    filename: str = None, 
                    fps: int = None,
                    export_settings: Dict[str, Any] = None) -> str:
        """
        Export a video to file with compatible settings.
        If video_clip is a file path (str), just move/copy it to the output directory.
        If video_clip is a VideoFileClip, use MoviePy export logic.
        """
        export_start = time.time()
        logger.info("Starting video export process (audio disabled)")
        
        # If video_clip is a file path, just move/copy it
        if isinstance(video_clip, str):
            # Generate output filename if not provided
            if not filename:
                timestamp = int(time.time())
                filename = f"export_{timestamp}"
            # Ensure filename has no extension
            filename = os.path.splitext(filename)[0]
            output_path = os.path.join(self.output_dir, f"{filename}.mp4")
            logger.info(f"Copying FFmpeg output file to: {output_path}")
            shutil.move(video_clip, output_path)
            logger.info(f"Exported file: {output_path}")
            return output_path
        
        # Handle VideoFileClip objects
        else:
            if video_clip is None:
                logger.error("Video clip is None, cannot export")
                raise ValueError("Video clip cannot be None")
                
            try:
                # First, ensure dimensions are even (required by H.264)
                width = video_clip.w
                height = video_clip.h
                
                # Round to nearest even number
                if width % 2 != 0:
                    width = (width // 2) * 2
                if height % 2 != 0:
                    height = (height // 2) * 2
                    
                # Resize if dimensions changed
                if width != video_clip.w or height != video_clip.h:
                    logger.info(f"Resizing video from {video_clip.w}x{video_clip.h} to {width}x{height} for codec compatibility")
                    video_clip = video_clip.resize(width=width, height=height)
                
                # Remove audio from clip to prevent 'NoneType' object has no attribute 'stdout' errors
                try:
                    video_clip = video_clip.without_audio()
                    logger.info("Audio removed from clip to prevent FFmpeg errors")
                except Exception as e:
                    logger.warning(f"Error removing audio from clip: {e}")
                
                # Default settings for maximum compatibility
                default_settings = {
                    'codec': 'libx264',
                    'preset': 'medium',  # Use medium preset for balance
                    'bitrate': '2000k',  # Lower bitrate for better compatibility
                    'threads': self.cpu_count,
                    'ffmpeg_params': [
                        '-an',  # Disable audio
                        '-pix_fmt', 'yuv420p',  # Standard pixel format
                        '-profile:v', 'baseline',  # Most compatible H.264 profile
                        '-level', '3.0',  # Compatible level
                        '-movflags', '+faststart',  # Enable fast start for web playback
                        '-vf', f'scale={width}:{height}'  # Force dimensions
                    ],
                    'verbose': False,
                    'logger': 'bar',
                    'audio': False,
                    'write_logfile': False
                }
                
                # Override with custom settings if provided, but keep essential compatibility settings
                if export_settings:
                    # Only update non-critical settings
                    for key in ['preset', 'bitrate', 'threads', 'verbose', 'logger']:
                        if key in export_settings:
                            default_settings[key] = export_settings[key]
                    
                # Generate output filename if not provided
                if not filename:
                    timestamp = int(time.time())
                    filename = f"export_{timestamp}"
                    
                # Ensure filename has no extension
                filename = os.path.splitext(filename)[0]
                output_path = os.path.join(self.output_dir, f"{filename}.mp4")
                
                # Set fps if provided, otherwise ensure it's a standard value
                if fps:
                    video_clip = video_clip.set_fps(fps)
                elif video_clip.fps > 60:
                    video_clip = video_clip.set_fps(60)  # Cap at 60fps for compatibility
                    
                # Log export settings
                logger.info(f"Exporting video: {output_path}")
                logger.info(f"Video duration: {video_clip.duration:.2f}s")
                logger.info(f"Video dimensions: {width}x{height}")
                logger.info(f"Video FPS: {video_clip.fps}")
                logger.info(f"Export settings: {default_settings}")
                
                # Use temp file for safer export
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                    temp_path = temp_file.name
                
                try:
                    # Export with compatible settings
                    video_clip.write_videofile(
                        temp_path,
                        fps=video_clip.fps,
                        codec=default_settings['codec'],
                        preset=default_settings['preset'],
                        bitrate=default_settings['bitrate'],
                        threads=default_settings['threads'],
                        ffmpeg_params=default_settings['ffmpeg_params'],
                        verbose=default_settings['verbose'],
                        logger=default_settings['logger'],
                        audio=False,
                        write_logfile=False
                    )
                    
                    # Move temp file to final destination
                    logger.info(f"Moving temp file to: {output_path}")
                    # Add a small delay to ensure file is fully written and released
                    time.sleep(0.2)
                    
                    # Try a few times to move the file in case of file locking issues
                    max_attempts = 3
                    for attempt in range(max_attempts):
                        try:
                            shutil.move(temp_path, output_path)
                            break
                        except PermissionError as e:
                            if attempt < max_attempts - 1:
                                logger.warning(f"File access error on attempt {attempt+1}, retrying in 0.5s: {e}")
                                time.sleep(0.5)  # Wait a bit longer between retries
                            else:
                                raise  # Re-raise if all attempts failed
                    
                except Exception as e:
                    logger.error(f"Export failed: {e}")
                    # Clean up the temp file
                    try:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                    except:
                        pass
                    raise
                    
                logger.info(f"Video export completed successfully in {time.time() - export_start:.2f}s")
                logger.info(f"Exported file: {output_path}")
                
                return output_path
                
            except Exception as e:
                logger.error(f"Error exporting video: {e}", exc_info=True)
                raise
            finally:
                # Clean up resources - ONLY for VideoFileClip objects!
                try:
                    if not isinstance(video_clip, str):
                        video_clip.close()
                except:
                    pass

    def batch_export(self, video_clips: list, original_filename: str,
                    metadata_list: Optional[list] = None) -> list:
        """Export multiple video variations."""
        output_paths = []
        for i, clip in enumerate(video_clips):
            metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else None
            output_path = self.export_video(
                clip,
                original_filename,
                variation=i,
                metadata=metadata
            )
            output_paths.append(output_path)
        return output_paths

    def cleanup(self, video_clip: VideoFileClip):
        """Clean up resources after export."""
        if video_clip:
            video_clip.close() 