import os
import argparse
import random
from src.video.input import VideoInput
from src.video.transform import VideoTransformer
from src.audio.processor import AudioProcessor
from src.video.export import VideoExporter
from src.audio.voice_over import VoiceOverGenerator

def setup_directories():
    """Create necessary directories if they don't exist."""
    directories = ['input', 'output', 'config']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def get_random_effects(video_clip, variation: int):
    """Generate random effects for a variation."""
    transformer = VideoTransformer(video_clip)
    
    # List of possible effects
    zoom_factors = [1.1, 1.2, 1.3, 1.4, 1.5]
    crop_percentages = [0.1, 0.15, 0.2]
    blur_intensities = [1.0, 1.5, 2.0, 2.5]
    color_intensities = [1.1, 1.2, 1.3, 1.4, 1.5]  # Lowered from 1.2-2.0
    filter_types = ['grayscale', 'blur', 'colorx', 'sepia', 'invert', 'brightness', 'contrast', 'saturation']
    hash_methods = ['pixel', 'delay', 'watermark']  # Added hash modification methods
    
    # Always apply a random hash modification
    transformer.modify_hash(random.choice(hash_methods))
    
    # Apply random effects based on variation number
    if variation % 5 == 0:  # Every 5th variation
        transformer.apply_zoom(random.choice(zoom_factors))
        transformer.apply_filter(random.choice(filter_types), intensity=random.choice(color_intensities))
    elif variation % 5 == 1:
        transformer.apply_crop(random.choice(crop_percentages))
        transformer.apply_filter(random.choice(filter_types), intensity=random.choice(color_intensities))
    elif variation % 5 == 2:
        transformer.apply_zoom(random.choice(zoom_factors))
        transformer.apply_filter(random.choice(filter_types), intensity=random.choice(color_intensities))
    elif variation % 5 == 3:
        transformer.apply_crop(0.1)  # Fixed crop percentage
        transformer.apply_filter(random.choice(filter_types), intensity=random.choice(color_intensities))
    else:  # variation % 5 == 4
        transformer.apply_zoom(random.choice(zoom_factors))
        transformer.apply_filter(random.choice(filter_types), intensity=random.choice(color_intensities))
        transformer.apply_filter(random.choice(filter_types), intensity=random.choice(color_intensities))
    
    # Add random transitions
    if random.random() < 0.5:
        transformer.apply_transition('fadein', duration=random.uniform(0.5, 1.5))
    if random.random() < 0.5:
        transformer.apply_transition('fadeout', duration=random.uniform(0.5, 1.5))
    
    return transformer

def get_random_audio_effects(video_clip, variation: int):
    """Generate random audio effects for a variation."""
    if video_clip.audio is None:
        return None
        
    audio_processor = AudioProcessor(video_clip)
    
    # Random volume adjustment
    volume_factor = random.uniform(0.8, 1.5)
    audio_processor.adjust_volume(volume_factor)
    
    # Random fade in/out
    if random.random() < 0.7:
        audio_processor.fade_in(random.uniform(0.5, 1.5))
    if random.random() < 0.7:
        audio_processor.fade_out(random.uniform(0.5, 1.5))
    
    return audio_processor

def process_video(input_path: str, variation: int = 0, add_voiceover: bool = False):
    """Process a single video file with transformations and effects."""
    try:
        with VideoInput(input_path) as video:
            # Analyze video properties
            properties = video.analyze()
            print(f"\nProcessing variation {variation + 1}/50")
            print(f"Video Properties:")
            for key, value in properties.items():
                print(f"{key}: {value}")
            
            # Initialize components
            transformer = get_random_effects(video.video_clip, variation)
            exporter = VideoExporter()
            
            # Get transformed clip
            transformed_clip = transformer.get_transformed_clip()

            # Process audio with random effects
            audio_processor = get_random_audio_effects(video.video_clip, variation)
            if audio_processor:
                modified_audio = audio_processor.get_modified_clip()
                transformed_clip = transformed_clip.set_audio(modified_audio)

            # Add AI voice-over if requested
            if add_voiceover:
                print("Generating AI voice-over...")
                voice_gen = VoiceOverGenerator()
                transformed_clip = voice_gen.add_to_video(
                    transformed_clip,
                    speed=random.uniform(0.9, 1.1),  # Slight speed variation
                    pitch=random.uniform(0.95, 1.05),  # Slight pitch variation
                    original_audio_volume=0.3
                )
                voice_gen.cleanup()

            # Prepare metadata
            metadata = {
                'original_file': input_path,
                'properties': properties,
                'effects': transformer.get_effects(),
                'has_audio': video.video_clip.audio is not None,
                'variation': variation,
                'has_voiceover': add_voiceover
            }

            # Export the processed video
            output_path = exporter.export_video(
                transformed_clip,
                os.path.basename(input_path),
                variation=variation,
                metadata=metadata
            )
            
            print(f"Processed video variation {variation} saved to: {output_path}")
            
            # Cleanup
            exporter.cleanup(transformed_clip)
            
    except Exception as e:
        print(f"Error processing video variation {variation}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Reels Editor - Video Automation Tool')
    parser.add_argument('--input', '-i', help='Input video file path')
    parser.add_argument('--batch', '-b', action='store_true', help='Process all videos in input directory')
    parser.add_argument('--variations', '-v', type=int, default=50, help='Number of variations to generate')
    parser.add_argument('--voiceover', action='store_true', help='Add AI voice-over to variations')
    
    args = parser.parse_args()
    
    # Setup project directories
    setup_directories()
    
    if args.batch:
        # Process all videos in input directory
        input_dir = 'input'
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.mp4', '.mov', '.avi')):
                input_path = os.path.join(input_dir, filename)
                print(f"\nProcessing: {filename}")
                for variation in range(args.variations):
                    process_video(input_path, variation, args.voiceover)
    elif args.input:
        # Process single video
        for variation in range(args.variations):
            process_video(args.input, variation, args.voiceover)
    else:
        print("Please provide an input video file or use --batch to process all videos in the input directory")
        print("Usage: python main.py --input video.mp4")
        print("       python main.py --batch")
        print("       python main.py --input video.mp4 --variations 50")

if __name__ == "__main__":
    main() 