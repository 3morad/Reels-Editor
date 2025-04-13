import streamlit as st
import os
from pathlib import Path
import tempfile
from src.video.input import VideoInput
from src.video.core.transformer import VideoTransformer
import random
import atexit
from videohash import VideoHash
import zipfile
import traceback
from datetime import datetime
from src.video.utils.logging_utils import configure_logger
import time

# Configure logger
logger = configure_logger("App")

# Set page config
st.set_page_config(
    page_title="Reels Editor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stFileUploader>div>div>div>div {
        background-color: #262730;
        border-radius: 5px;
    }
    .css-1d391kg {
        background-color: #262730;
    }
    .effects-info {
        background-color: #262730;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
    }
    .effect-section {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🎬 Reels Editor")
st.markdown("""
    Transform your videos into unique social media reels with AI-powered effects and variations.
    Upload your video and customize the effects to create stunning content!
    """)

# Store temporary files to clean up later
temp_files = []

def cleanup_temp_files():
    """Clean up all temporary files."""
    for file_path in temp_files:
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as e:
            st.warning(f"Could not delete temporary file {file_path}: {str(e)}")

# Register cleanup function
atexit.register(cleanup_temp_files)

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Video upload
    uploaded_file = st.file_uploader("Upload your video", type=['mp4', 'mov', 'avi'])
    
    # Number of variations
    num_variations = st.slider("Number of Variations", 1, 50, 1, 
                             help="Select how many different variations to generate")
    
    # Effect selection
    st.subheader("Select Effects to Apply")
    effects = {
        "Zoom": st.checkbox("Zoom", value=True),
        "Crop": st.checkbox("Crop", value=True),
        "Filter": st.checkbox("Filter", value=True),
        "Transition": st.checkbox("Transition", value=True),
        "Trim": st.checkbox("Trim", value=False),
        "Hash Modification": st.checkbox("Hash Modification", value=True)
    }
    
    # Hash modification options (only show if Hash Modification is selected)
    if effects["Hash Modification"]:
        st.subheader("Hash Modification Options")
        hash_methods = [
            'pixelate',  # Basic pixel-level modifications
            'glitch',  # Glitch effect with horizontal shifts
            'dct',  # Frequency domain modifications
            'delay',  # Frame delay/insertion
            'watermark',  # Pattern-based watermarking
            'noise',  # Selective noise addition
            'color',  # Color space manipulations
            'metadata'  # Metadata modifications
        ]
        
        # Select all methods by default
        selected_methods = st.multiselect(
            "Select Hash Modification Methods",
            hash_methods,
            default=hash_methods,  # All methods selected by default
            help="Choose one or more methods to modify the video hash while preserving visual quality"
        )
        
        # Add descriptions for each method
        method_descriptions = {
            'pixelate': "Basic pixel-level modifications with subtle brightness and noise changes",
            'glitch': "Applies glitch effect with horizontal shifts",
            'dct': "Applies frequency domain modifications using DCT coefficients",
            'delay': "Inserts imperceptible frame delays at the beginning of the video",
            'watermark': "Adds a subtle pattern-based watermark that affects hash generation",
            'noise': "Adds calibrated noise patterns below human perception threshold",
            'color': "Modifies color space components while preserving visual appearance",
            'metadata': "Applies subtle modifications to video metadata (FPS, resolution, duration)"
        }
        
        # Display descriptions for selected methods
        for method in selected_methods:
            st.info(f"**{method}**: {method_descriptions[method]}")
    
    # System requirements note
    st.sidebar.markdown("""
    ### System Requirements
    - Tesseract OCR must be installed for text relocation features
    - FFmpeg must be installed for video processing
    - Python 3.8+ required
    """)
    
    # Processing options
    st.subheader("Processing Options")
    sample_rate = st.slider("Frame Sample Rate", 1, 60, 30)
    quality = st.select_slider("Output Quality", options=["Low", "Medium", "High"], value="Medium")

def process_video(uploaded_file, num_variations=3, apply_random_effects=True):
    """Process the uploaded video file and generate variations"""
    try:
        # Create a temporary file to store the uploaded video
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Register the temporary file for cleanup
        temp_files.append(tmp_path)
        
        # Initialize video input
        video_input = VideoInput(tmp_path)
        
        # Get video properties
        properties = video_input.analyze()
        
        # Create transformer
        transformer = VideoTransformer(video_input.video_clip)
        
        # Apply random effects if requested
        if apply_random_effects:
            # Apply zoom effect
            zoom_factor = random.uniform(1.05, 1.15)
            transformer.apply_zoom(zoom_factor)
            
            # Apply crop effect
            crop_percent = random.uniform(0.05, 0.2)
            transformer.apply_crop(crop_percent)
            
            # Apply filter effect
            filter_types = ["brightness"]
            filter_type = random.choice(filter_types)
            # Ensure intensity is between 0 and 1, but use much lower values
            intensity = random.uniform(0.05, 0.2)  # Reduced from 0.1-0.9 to 0.05-0.2
            transformer.apply_filter(filter_type, intensity)
            
            # Apply transition effect
            transition_types = ["fadein", "fadeout"]
            transition_type = random.choice(transition_types)
            duration = random.uniform(0.5, 1.5)
            transformer.apply_transition(transition_type, duration)
        
        # Generate variations
        variations = []
        for i in range(num_variations):
            # Apply hash modification
            hash_types = ["pixelate", "glitch", "metadata"]  # Added metadata to default types
            hash_type = random.choice(hash_types)
            # Ensure intensity is between 0 and 1
            intensity = random.uniform(0.1, 0.3)  # Reduced intensity for more subtle changes
            
            # Apply hash modification using the hash_effects module
            transformer.modify_hash(hash_type, intensity)
            
            # Get transformed clip
            transformed_clip = transformer.get_transformed_clip()
            
            # Create a temporary file for the variation
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as var_file:
                var_path = var_file.name
            
            # Register the variation file for cleanup
            temp_files.append(var_path)
            
            # Write the variation to the temporary file
            transformed_clip.write_videofile(var_path, codec='libx264', audio_codec='aac')
            
            # Add the variation to the list
            variations.append({
                'path': var_path,
                'effects': transformer.get_effects()
            })
            
            # Reset the transformer for the next variation
            transformer.reset()
        
        # Close the video input
        video_input.close()
        
        return properties, variations
    
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"Error processing video: {str(e)}")
        return None, None

# Main content area
if uploaded_file is not None:
    # Save uploaded file to temp directory
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        video_path = tmp_file.name
        temp_files.append(video_path)

    # Display video preview
    st.video(video_path)
    
    # Process button
    if st.button("Process Video", key="process_button"):
        try:
            # Create output directory if it doesn't exist
            os.makedirs("output", exist_ok=True)
            
            # Initialize video input with validation
            video_input = VideoInput(video_path)
            
            # Get hash of original video
            original_hash = VideoHash(video_path)
            st.info(f"Original Video Hash: {original_hash.hash_hex}")
            
            # Analyze video properties
            video_properties = video_input.analyze()
            st.info(f"Video Properties: Duration: {video_properties['duration']:.2f}s, Resolution: {video_properties['width']}x{video_properties['height']}, FPS: {video_properties['fps']}")
            
            # Process multiple variations
            processed_videos = []
            for i in range(num_variations):
                with st.spinner(f"Processing variation {i+1}/{num_variations}..."):
                    # Reset transformer for each variation
                    transformer = VideoTransformer(video_input.video_clip)
                    
                    # Apply selected effects
                    applied_effects = []
                    if effects["Zoom"]:
                        zoom_factor = random.uniform(1.05, 1.15)
                        transformer.apply_zoom(zoom_factor)
                        applied_effects.append(f"Zoom: {zoom_factor:.2f}x")
                    if effects["Crop"]:
                        crop_percent = random.uniform(0.1, 0.3)
                        transformer.apply_crop(crop_percent)
                        applied_effects.append(f"Crop: {crop_percent*100:.1f}%")
                    if effects["Filter"]:
                        # Only use brightness filter
                        filter_type = "brightness"
                        # Ensure intensity is between 0 and 1, but use much lower values
                        intensity = random.uniform(0.05, 0.2)  # Reduced from 0.1-0.9 to 0.05-0.2
                        transformer.apply_filter(filter_type, intensity)
                        applied_effects.append(f"Filter: {filter_type} ({intensity:.2f})")
                    if effects["Transition"]:
                        transition_type = random.choice(["fadein", "fadeout"])
                        duration = random.uniform(0.5, 2.0)
                        transformer.apply_transition(transition_type, duration)
                        applied_effects.append(f"Transition: {transition_type} ({duration:.1f}s)")
                    
                    if effects["Trim"]:
                        trim_percent = random.uniform(0.1, 0.3)  # Trim 10-30% from the end
                        transformer.apply_trim(trim_percent)
                        applied_effects.append(f"Trim: {trim_percent*100:.1f}% from end")
                    
                    if effects["Hash Modification"]:
                        # Apply all selected hash modification methods
                        for hash_method in selected_methods:
                            # Use higher intensity values for hash modifications
                            # This will ensure the changes are effective while still being visually subtle
                            intensity = random.uniform(0.08, 0.12)  # Increased from 0.03-0.05 to 0.08-0.12
                            transformer.modify_hash(hash_method, intensity)
                            applied_effects.append(f"Hash Modification: {hash_method} ({intensity:.3f})")
                    
                    # Generate output filename
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_filename = f"output/variation_{i+1}_{timestamp}.mp4"
                    
                    # Get transformed clip and export
                    transformed_clip = transformer.get_transformed_clip()
                    transformed_clip.write_videofile(
                        output_filename,
                        codec='libx264',
                        audio_codec='aac',
                        bitrate='5000k',
                        threads=4,
                        preset='medium'
                    )
                    processed_videos.append(output_filename)
                    
                    # Ensure the file is fully written and closed
                    transformed_clip.close()
                    
                    # Display video properties and hash information
                    st.subheader(f"Variation {i+1} Information")
                    st.write(f"Applied Effects: {', '.join(applied_effects)}")
                    
                    # Calculate and display hash information
                    try:
                        # Add a small delay to ensure file is fully written
                        time.sleep(0.5)
                        
                        # Calculate hash with explicit file path
                        variation_hash = VideoHash(output_filename)
                        
                        # Calculate hash difference
                        # Convert hex strings to integers first
                        original_hash_int = int(original_hash.hash_hex, 16)
                        variation_hash_int = int(variation_hash.hash_hex, 16)
                        
                        # Calculate Hamming distance
                        hash_diff = bin(original_hash_int ^ variation_hash_int).count('1')
                        hash_diff_percent = (hash_diff / 64) * 100  # 64 bits in the hash
                        
                        st.write("Hash Information:")
                        st.write(f"Original Hash: {original_hash.hash_hex}")
                        st.write(f"Variation Hash: {variation_hash.hash_hex}")
                        st.write(f"Hash Difference: {hash_diff_percent:.2f}%")
                    except Exception as e:
                        st.error(f"Error calculating hash: {e}")
                        print(f"Hash calculation error: {e}")
                        traceback.print_exc()
                    
                    # Display the processed video
                    st.video(output_filename)
            
            # Add download all variations button
            if processed_videos:
                st.subheader("Download All Variations")
                zip_filename = f"output/variations_{timestamp}.zip"
                with zipfile.ZipFile(zip_filename, 'w') as zipf:
                    for video in processed_videos:
                        zipf.write(video, os.path.basename(video))
                
                with open(zip_filename, 'rb') as f:
                    st.download_button(
                        label="Download All Variations",
                        data=f,
                        file_name=f"variations_{timestamp}.zip",
                        mime="application/zip"
                    )
            
            # Clean up video input
            video_input.close()
                
        except Exception as e:
            st.error(f"Error processing video: {e}")
            print(f"ERROR processing video: {e}")
            traceback.print_exc()
            # Ensure cleanup even if there's an error
            if 'video_input' in locals():
                video_input.close()
else:
    st.info("👆 Upload a video to get started!")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Made with ❤️ using Streamlit</p>
    </div>
    """, unsafe_allow_html=True) 