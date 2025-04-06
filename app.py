import streamlit as st
import os
from pathlib import Path
import tempfile
from src.video.input import VideoInput
from src.video.transform import VideoTransformer
import random
import atexit
from videohash import VideoHash

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
    num_variations = st.slider("Number of variations", 1, 10, 3)
    
    # Effect selection
    st.subheader("Effects")
    effects = {
        "Zoom": st.checkbox("Random Zoom", value=True),
        "Crop": st.checkbox("Random Crop", value=True),
        "Filter": st.checkbox("Random Filter", value=True),
        "Transition": st.checkbox("Random Transition", value=True),
        "Text Relocation": st.checkbox("Text Relocation", value=True),
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
            'geometric',  # Spatial transformations
            'color'  # Color space manipulations
        ]
        
        # Use multiselect instead of selectbox
        selected_methods = st.multiselect(
            "Select Hash Modification Methods",
            hash_methods,
            default=['pixel', 'delay', 'watermark'],  # Default selection
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
            'geometric': "Applies subtle perspective and spatial transformations",
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
    if st.button("✨ Generate Variations"):
        with st.spinner("Processing your video..."):
            try:
                # Initialize video input with validation
                video_input = VideoInput(video_path)
                
                # Get hash of original video
                original_hash = VideoHash(video_path)
                st.info(f"Original Video Hash: {original_hash.hash_hex}")
                
                # Analyze video properties
                video_properties = video_input.analyze()
                st.info(f"Video Properties: Duration: {video_properties['duration']:.2f}s, Resolution: {video_properties['width']}x{video_properties['height']}, FPS: {video_properties['fps']}")
                
                # Create output directory if it doesn't exist
                output_dir = Path("output")
                output_dir.mkdir(exist_ok=True)
                
                # Process variations
                for i in range(num_variations):
                    # Create transformer with the video clip
                    transformer = VideoTransformer(video_input.video_clip)
                    
                    # Apply random effects
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
                    
                    if effects["Text Relocation"]:
                        transformer.relocate_text()
                        applied_effects.append("Text Relocation")
                    
                    if effects["Hash Modification"]:
                        # Apply all selected hash modification methods
                        for hash_method in selected_methods:
                            transformer.modify_hash(hash_method)
                            applied_effects.append(f"Hash Modification: {hash_method}")
                    
                    # Get transformed clip and validate
                    transformed_clip = transformer.get_transformed_clip()
                    if not transformed_clip or not hasattr(transformed_clip, 'duration'):
                        raise ValueError("Failed to create transformed video clip")
                    
                    # Save variation
                    output_path = output_dir / f"variation_{i+1}.mp4"
                    transformed_clip.write_videofile(str(output_path))
                    
                    # Get hash of variation
                    variation_hash = VideoHash(str(output_path))
                    # Calculate hash difference using hamming distance
                    hash_difference = bin(int(original_hash.hash_hex, 16) ^ int(variation_hash.hash_hex, 16)).count('1') / 64.0
                    
                    # Display variation with effects info
                    st.subheader(f"Variation {i+1}")
                    st.video(str(output_path))
                    
                    # Display hash information
                    st.markdown(f"""
                        <div style='background-color: #1E1E1E; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                            <h4>Hash Information:</h4>
                            <p>Original Hash: {original_hash.hash_hex}</p>
                            <p>Variation Hash: {variation_hash.hash_hex}</p>
                            <p>Hash Difference: {hash_difference * 100:.2f}%</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Display applied effects
                    st.markdown("""
                        <div class="effects-info">
                            <h4>Applied Effects:</h4>
                            <ul>
                    """, unsafe_allow_html=True)
                    
                    for effect in applied_effects:
                        st.markdown(f"<li>{effect}</li>", unsafe_allow_html=True)
                    
                    st.markdown("</ul></div>", unsafe_allow_html=True)
                    
                    st.download_button(
                        label=f"Download Variation {i+1}",
                        data=open(output_path, "rb").read(),
                        file_name=f"variation_{i+1}.mp4",
                        mime="video/mp4"
                    )
                
                st.success("🎉 All variations generated successfully!")
                
            except Exception as e:
                st.error(f"Error processing video: {str(e)}")
            finally:
                # Close the video input to free resources
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