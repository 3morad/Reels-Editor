import os
import json
from datetime import datetime
from typing import Dict, Optional
from moviepy.editor import VideoFileClip

class VideoExporter:
    def __init__(self, output_dir: str = 'output'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

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

    def export_video(self, video_clip: VideoFileClip, original_filename: str,
                    variation: int = 0, metadata: Optional[Dict] = None,
                    codec: str = 'libx264', audio_codec: str = 'aac',
                    bitrate: str = '5000k') -> str:
        """Export the video with specified parameters."""
        # Generate output filename
        output_filename = self.generate_filename(original_filename, variation)
        output_path = os.path.join(self.output_dir, output_filename)

        # Export video
        video_clip.write_videofile(
            output_path,
            codec=codec,
            audio_codec=audio_codec,
            bitrate=bitrate,
            threads=4,
            preset='medium'
        )

        # Save metadata if provided
        if metadata:
            metadata['export_info'] = {
                'timestamp': datetime.now().isoformat(),
                'output_path': output_path,
                'codec': codec,
                'audio_codec': audio_codec,
                'bitrate': bitrate
            }
            self.save_metadata(metadata, output_filename)

        return output_path

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