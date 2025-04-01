from moviepy.editor import ColorClip
import numpy as np

def create_test_video():
    # Create a 10-second video
    duration = 10
    width, height = 1080, 1920  # Vertical video format for reels
    
    # Create a simple color clip (blue background)
    video = ColorClip(size=(width, height), color=(0, 0, 255))
    video = video.set_duration(duration)
    
    # Create input directory if it doesn't exist
    import os
    os.makedirs('input', exist_ok=True)
    
    # Write the video file
    output_path = 'input/test_video.mp4'
    video.write_videofile(
        output_path,
        fps=30,
        codec='libx264',
        audio=False,
        preset='medium'
    )
    
    # Clean up
    video.close()
    
    print(f"Test video created successfully at: {output_path}")

if __name__ == "__main__":
    create_test_video() 