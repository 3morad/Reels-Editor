import os
import json
from datetime import datetime
from typing import Dict, Optional, Any
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
import time
import logging
import multiprocessing

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
                    video_clip: VideoFileClip, 
                    audio_clip: AudioFileClip = None, 
                    filename: str = None, 
                    fps: int = None,
                    export_settings: Dict[str, Any] = None) -> str:
        """
        Export a video to file with optimized settings.
        
        Args:
            video_clip: The video clip to export
            audio_clip: Optional audio clip to add (ignored to prevent FFmpeg errors)
            filename: Output filename (without extension)
            fps: Frames per second
            export_settings: Dictionary with export settings
            
        Returns:
            Path to the exported video file
        """
        export_start = time.time()
        logger.info("Starting video export process (audio disabled)")
        
        if video_clip is None:
            logger.error("Video clip is None, cannot export")
            raise ValueError("Video clip cannot be None")
            
        try:
            # Remove audio from clip to prevent 'NoneType' object has no attribute 'stdout' errors
            try:
                video_clip = video_clip.without_audio()
                logger.info("Audio removed from clip to prevent FFmpeg errors")
            except Exception as e:
                logger.warning(f"Error removing audio from clip: {e}")
            
            # Default settings for optimal performance - audio disabled
            default_settings = {
                'codec': 'libx264',
                'preset': 'ultrafast',  # Options: ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow
                'bitrate': '8000k',
                # No audio settings
                'threads': self.cpu_count,
                'ffmpeg_params': ['-an','auto'],  # -an disables audio, enable hardware acceleration
                'verbose': False,
                'logger': 'bar',  # Options: bar, None
                # Explicitly disable audio
                'audio': False,
            }
            
            # Override with custom settings if provided
            if export_settings:
                default_settings.update(export_settings)
                # Force audio off regardless of custom settings
                default_settings['audio'] = False
                if '-an' not in default_settings.get('ffmpeg_params', []):
                    default_settings['ffmpeg_params'] = default_settings.get('ffmpeg_params', []) + ['-an']
                
            # Generate output filename if not provided
            if not filename:
                timestamp = int(time.time())
                filename = f"export_{timestamp}"
                
            # Ensure filename has no extension
            filename = os.path.splitext(filename)[0]
            output_path = os.path.join(self.output_dir, f"{filename}.mp4")
            
            # Set fps if provided
            if fps:
                video_clip = video_clip.set_fps(fps)
                
            # Audio is ignored - log this
            if audio_clip:
                logger.info("Audio clip provided but ignored to prevent FFmpeg errors")
                    
            # Log export settings
            logger.info(f"Exporting video: {output_path}")
            logger.info(f"Video duration: {video_clip.duration:.2f}s")
            logger.info(f"Video dimensions: {video_clip.w}x{video_clip.h}")
            logger.info(f"Video FPS: {video_clip.fps}")
            logger.info(f"Export settings: {default_settings}")
            
            # Use temp file for safer export
            import tempfile
            import shutil
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                temp_path = temp_file.name
            
            try:
                # First try with hardware acceleration
                logger.info(f"Exporting to temp file: {temp_path}")
                video_clip.write_videofile(
                    temp_path,
                    fps=video_clip.fps if not fps else fps,
                    codec=default_settings['codec'],
                    preset=default_settings['preset'],
                    bitrate=default_settings['bitrate'],
                    threads=default_settings['threads'],
                    ffmpeg_params=default_settings['ffmpeg_params'],
                    verbose=default_settings['verbose'],
                    logger=default_settings['logger'],
                    audio=False  # Explicitly disable audio
                )
                
                # Move temp file to final destination
                logger.info(f"Moving temp file to: {output_path}")
                shutil.move(temp_path, output_path)
                
            except Exception as e:
                logger.warning(f"First export attempt failed, trying simpler settings: {e}")
                
                # Clean up the temp file if it exists
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except:
                    pass
                
                # Create a new temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                    temp_path = temp_file.name
                
                # Fall back to CPU-only encoding with ultrasimple settings
                default_settings['ffmpeg_params'] = ['-an']  # Audio disabled
                default_settings['preset'] = 'ultrafast'  # Fastest encoding
                
                try:
                    video_clip.write_videofile(
                        temp_path,
                        fps=video_clip.fps if not fps else fps,
                        codec=default_settings['codec'],
                        preset=default_settings['preset'],
                        bitrate=default_settings['bitrate'],
                        threads=default_settings['threads'],
                        ffmpeg_params=default_settings['ffmpeg_params'],
                        verbose=default_settings['verbose'],
                        logger=default_settings['logger'],
                        audio=False  # Explicitly disable audio
                    )
                    
                    # Move temp file to final destination
                    shutil.move(temp_path, output_path)
                    
                except Exception as e2:
                    logger.error(f"All export attempts failed: {e2}")
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
            # No audio resources to clean up
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