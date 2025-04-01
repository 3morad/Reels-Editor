import cv2
import numpy as np
from src.video.text_processor import TextProcessor
from moviepy.editor import VideoFileClip
import argparse

def test_single_frame(image_path):
    """Test text detection on a single image."""
    print(f"\n=== Testing Text Detection on Image: {image_path} ===")
    
    # Read the image
    frame = cv2.imread(image_path)
    if frame is None:
        print("ERROR: Could not read image file")
        return
        
    # Initialize text processor
    text_processor = TextProcessor()
    
    # Detect text regions
    text_regions = text_processor.detect_text_regions(frame)
    
    # Draw detected regions on image
    output = frame.copy()
    for region in text_regions:
        # Draw rectangle
        cv2.rectangle(output, 
                     (region.x, region.y), 
                     (region.x + region.width, region.y + region.height), 
                     (0, 255, 0), 2)
        # Draw text
        cv2.putText(output, 
                    f"{region.text} ({region.confidence:.2f})", 
                    (region.x, region.y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Save output image
    output_path = "text_detection_result.jpg"
    cv2.imwrite(output_path, output)
    print(f"Results saved to: {output_path}")
    print(f"Found {len(text_regions)} text regions:")
    for region in text_regions:
        print(f"Text: {region.text}")
        print(f"Position: ({region.x}, {region.y})")
        print(f"Size: {region.width}x{region.height}")
        print(f"Confidence: {region.confidence}")
        print("---")

def test_video_frames(video_path, sample_rate=1):
    """Test text detection on video frames."""
    print(f"\n=== Testing Text Detection on Video: {video_path} ===")
    
    # Load video
    video = VideoFileClip(video_path)
    if video is None:
        print("ERROR: Could not load video file")
        return
        
    # Initialize text processor
    text_processor = TextProcessor()
    
    # Process frames
    frame_count = 0
    detected_count = 0
    
    def process_frame(frame):
        nonlocal frame_count, detected_count
        
        # Only process every nth frame
        if frame_count % sample_rate != 0:
            frame_count += 1
            return frame
            
        print(f"\nProcessing frame {frame_count}")
        
        try:
            # Ensure frame is a numpy array
            if not isinstance(frame, np.ndarray):
                print("ERROR: Frame is not a numpy array")
                frame_count += 1
                return frame
                
            # Convert frame to BGR (MoviePy uses RGB)
            frame_bgr = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            # Detect text regions
            text_regions = text_processor.detect_text_regions(frame_bgr)
            
            if text_regions:
                detected_count += 1
                print(f"Found {len(text_regions)} text regions:")
                for region in text_regions:
                    print(f"Text: {region.text}")
                    print(f"Confidence: {region.confidence}")
                
                # Draw detected regions
                for region in text_regions:
                    cv2.rectangle(frame, 
                                (region.x, region.y), 
                                (region.x + region.width, region.y + region.height), 
                                (0, 255, 0), 2)
                    cv2.putText(frame, 
                               region.text, 
                               (region.x, region.y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                               
                # Save frame with detections
                output_path = f"text_detection_frame_{frame_count}.jpg"
                cv2.imwrite(output_path, cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR))
                print(f"Frame saved to: {output_path}")
            
        except Exception as e:
            print(f"ERROR processing frame {frame_count}: {e}")
        
        frame_count += 1
        return frame
    
    # Process video
    try:
        # Use fl_image instead of fl
        video.fl_image(process_frame)
        print(f"\nProcessing complete!")
        print(f"Total frames processed: {frame_count}")
        print(f"Frames with text detected: {detected_count}")
    except Exception as e:
        print(f"Error processing video: {e}")
    finally:
        video.close()

def main():
    parser = argparse.ArgumentParser(description='Test Text Detection')
    parser.add_argument('--input', '-i', required=True, help='Input image or video file')
    parser.add_argument('--type', '-t', choices=['image', 'video'], required=True, 
                       help='Input type (image or video)')
    parser.add_argument('--sample-rate', '-s', type=int, default=30,
                       help='Process every nth frame (video only)')
    
    args = parser.parse_args()
    
    if args.type == 'image':
        test_single_frame(args.input)
    else:
        test_video_frames(args.input, args.sample_rate)

if __name__ == "__main__":
    main() 