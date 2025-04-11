import streamlit as st
import os
from pathlib import Path
import tempfile
from src.video.input import VideoInput
from src.video.transform import VideoTransformer
import random
import atexit
from videohash import VideoHash
import zipfile
import traceback
from datetime import datetime

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
    st.subheader("Effects")
    effects = {
        "Zoom": st.checkbox("Random Zoom", value=True),
        "Crop": st.checkbox("Random Crop", value=True),
        "Filter": st.checkbox("Random Filter", value=True),
        "Transition": st.checkbox("Random Transition", value=True),
        # "Text Relocation": st.checkbox("Text Relocation", value=True),  # Hidden for now
        "Hash Modification": st.checkbox("Hash Modification", value=True)
    }
    
    # Hash modification options (only show if Hash Modification is selected)
    if effects["Hash Modification"]:
        st.subheader("Hash Modification Options")
        hash_methods = [
            'pixel',  # Basic pixel-level modifications
            'delay',  # Frame delay/insertion
            'watermark',  # Pattern-based watermarking
            'dct',  # Frequency domain modifications
            'temporal',  # Frame pattern modifications
            'noise',  # Selective noise addition
            'color'  # Color space manipulations
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
            'pixel': "Basic pixel-level modifications with subtle brightness and noise changes",
            'delay': "Inserts imperceptible frame delays at the beginning of the video",
            'watermark': "Adds a subtle pattern-based watermark that affects hash generation",
            'dct': "Applies frequency domain modifications using DCT coefficients",
            'temporal': "Modifies frame patterns and timing while preserving visual flow",
            'noise': "Adds calibrated noise patterns below human perception threshold",
            'color': "Modifies color space components while preserving visual appearance"
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
                        zoom_factor = random.uniform(1.1, 1.5)
                        transformer.apply_zoom(zoom_factor)
                        applied_effects.append(f"Zoom: {zoom_factor:.2f}x")
                    if effects["Crop"]:
                        crop_percent = random.uniform(0.1, 0.3)
                        transformer.apply_crop(crop_percent)
                        applied_effects.append(f"Crop: {crop_percent*100:.1f}%")
                    if effects["Filter"]:
                        filter_type = random.choice(["grayscale", "blur", "colorx", "sepia", "invert", "brightness", "contrast", "saturation"])
                        intensity = random.uniform(0.5, 1.5)
                        transformer.apply_filter(filter_type, intensity)
                        applied_effects.append(f"Filter: {filter_type} ({intensity:.2f})")
                    if effects["Transition"]:
                        transition_type = random.choice(["fadein", "fadeout"])
                        duration = random.uniform(0.5, 2.0)
                        transformer.apply_transition(transition_type, duration)
                        applied_effects.append(f"Transition: {transition_type} ({duration:.1f}s)")
                    # if effects["Text Relocation"]:  # Hidden for now
                    #     transformer.relocate_text()
                    #     applied_effects.append("Text Relocation")
                    
                    if effects["Hash Modification"]:
                        # Apply all selected hash modification methods
                        for hash_method in selected_methods:
                            transformer.modify_hash(hash_method, min_difference=0.05, variation_id=i)
                            applied_effects.append(f"Hash Modification: {hash_method}")
                    
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
                    
                    # Display video properties and hash information
                    st.subheader(f"Variation {i+1} Information")
                    st.write(f"Applied Effects: {', '.join(applied_effects)}")
                    
                    # Calculate and display hash information
                    try:
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
                    
                    # Clean up the transformed clip
                    transformed_clip.close()
            
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