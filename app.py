import streamlit as st
import os
import tempfile
import random
import atexit
import zipfile
import traceback
from datetime import datetime
import time
import logging

# Optional garbage collection
try:
    import gc
    gc.enable()
except ImportError:
    pass

from concurrent.futures import ThreadPoolExecutor 
from src.video.input import VideoInput
from src.video.core.transformer import VideoTransformer
from src.video.utils.logging_utils import configure_logger
from src.video.utils.hash_calculator import calculate_video_hash, calculate_video_difference
from src.video.effects.hash_presets import HASH_PRESETS, get_preset_default_intensity
from src.video.export import VideoExporter

# Configure logger
logger = configure_logger("App")
# Enable DEBUG level to see all debug logs
logger.setLevel(logging.DEBUG)

# Default configuration settings for batch processing
CONFIG = {
    "batch_size": 5,               # Number of variations to process at once
    "max_workers": 4,              # Maximum number of parallel processes
    "batch_cooldown_seconds": 2,   # Seconds to wait between batches
}

# Exporter and temp file tracking
exporter = VideoExporter(output_dir="output")
temp_files: list[str] = []

def cleanup_temp_files():
    for fp in temp_files:
        try:
            os.remove(fp)
        except Exception:
            pass
atexit.register(cleanup_temp_files)

# Top-level variation worker (must be pickle-friendly)
def _process_variation(args):
    tmp_path, i, effects, selected_preset, selected_methods, quality, fps = args
    try:
        vi = VideoInput(tmp_path)
        tf = VideoTransformer(vi.video_clip)
        # Apply effects based on selection
        if effects.get("Zoom"): tf.apply_zoom(random.uniform(1.05, 1.15))
        if effects.get("Crop"): tf.apply_crop(random.uniform(0.05, 0.2))
        if effects.get("Filter"): tf.apply_filter(random.choice(["brightness", "contrast", "saturation"]), random.uniform(0.1, 0.3))
        if effects.get("Transition"): tf.apply_transition(random.choice(["fadein", "fadeout"]), random.uniform(0.5, 1.5))
        if effects.get("Trim"): tf.apply_trim(random.uniform(0.1, 0.3))
        if effects.get("Hash Modification"):
            for ht in selected_methods:
                base = get_preset_default_intensity(selected_preset, ht)
                inten = max(0.1, min(1.0, base * random.uniform(0.8, 1.2)))
                tf.modify_hash(ht, inten)
        # Bake transformations
        clip = tf.get_transformed_clip()
        fname = f"variation_{i+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        out = exporter.export_video(
            video_clip=clip,
            filename=fname,
            fps=fps,
            export_settings={
                'preset': 'ultrafast' if quality == 'Low' else 'medium' if quality == 'Medium' else 'slow',
                'ffmpeg_params': ['-an', '-crf', '18']
            }
        )
        # Build effect string list
        eff_strs: list[str] = []
        for eff in tf.get_effects():
            t = eff.get('type', '')
            params = ", ".join(
                f"{k}={v:.2f}" if isinstance(v, (int, float)) else f"{k}={v}"
                for k, v in eff.items() if k != 'type'
            )
            eff_strs.append(f"{t}({params})" if params else t)
        # Cleanup
        tf.reset()
        vi.close()
        clip.close()
        return out, eff_strs
    except Exception as e:
        # Log the error and return empty results
        error_msg = f"Error processing variation {i+1}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return None, [f"Error: {str(e)}"]

# Orchestrate parallel processing with batching
def process_video(uploaded_file, num_variations, effects, selected_preset, selected_methods, quality, sample_rate):
    # Save upload to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    
    # DO NOT add to temp_files list - we'll handle cleanup explicitly at the end
    
    # Analyze properties
    vi = VideoInput(tmp_path)
    props = vi.analyze()
    fps = props['fps']
    vi.close()
    
    # Prepare arguments for each variation
    tasks = [
        (tmp_path, i, effects, selected_preset, selected_methods, quality, fps)
        for i in range(num_variations)
    ]
    
    # Process in batches to avoid resource exhaustion
    results = []
    batch_size = CONFIG["batch_size"]
    max_workers = min(CONFIG["max_workers"], exporter.cpu_count)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for batch_idx in range(0, len(tasks), batch_size):
        batch = tasks[batch_idx:batch_idx + batch_size]
        status_text.text(f"Processing batch {batch_idx//batch_size + 1}/{(len(tasks) + batch_size - 1)//batch_size}...")
        
        batch_results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for out, effs in pool.map(_process_variation, batch):
                if out is not None:  # Only add successful variations
                    batch_results.append({'path': out, 'effects': effs})
                
        results.extend(batch_results)
        
        # Update progress
        progress = min(1.0, (batch_idx + len(batch)) / len(tasks))
        progress_bar.progress(progress)
        
        # Force garbage collection between batches
        gc.collect()
        
        # Small delay to let system resources recover
        if batch_idx + batch_size < len(tasks):  # If not the last batch
            time.sleep(CONFIG["batch_cooldown_seconds"])
    
    progress_bar.progress(1.0)
    status_text.text("Processing complete!")
    
    return props, results, tmp_path  # Return the temp path for hash calculation

# Function to add advanced settings
def add_advanced_settings(sidebar):
    with sidebar.expander("🔧 Advanced Settings"):
        st.caption("These settings control resource usage")
        CONFIG["batch_size"] = st.slider("Batch Size", 1, 10, CONFIG["batch_size"], 
                                        help="Number of variations to process at once")
        CONFIG["max_workers"] = st.slider("Max Workers", 1, min(8, exporter.cpu_count), CONFIG["max_workers"], 
                                         help="Maximum number of parallel processes")
        CONFIG["batch_cooldown_seconds"] = st.slider("Batch Cooldown (s)", 0, 10, CONFIG["batch_cooldown_seconds"], 
                                                   help="Seconds to wait between batches")

# Main Streamlit application
def main():
    # Page config
    st.set_page_config(
        page_title="Reels Editor",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    # CSS
    st.markdown(
        """
        <style>
        .main { background-color: #0E1117; color: #FAFAFA; }
        .stButton>button { background-color: #FF4B4B; color: white; border-radius: 5px; padding: 10px 20px; font-weight: bold; }
        .stFileUploader>div>div>div>div { background-color: #262730; border-radius: 5px; }
        .css-1d391kg { background-color: #262730; }
        .effects-info { background-color: #262730; padding: 10px; border-radius: 5px; margin-top: 10px; }
        .effect-section { background-color: #1E1E1E; padding: 15px; border-radius: 5px; margin-bottom: 10px; }
        </style>
        """,
        unsafe_allow_html=True
    )
    # Title
    st.title("🎬 Reels Editor")
    st.markdown(
        """
        Transform your videos into unique social media reels with AI-powered effects and variations.
        Upload your video and customize the effects to create stunning content!
        """
    )

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Settings")
        uploaded_file = st.file_uploader("Upload your video", type=['mp4', 'mov', 'avi'])
        num_variations = st.slider("Number of Variations", 1, 65, 1)
        st.subheader("Select Effects to Apply")
        effects = {k: st.checkbox(k, value=(k != 'Trim')) for k in ["Zoom", "Crop", "Filter", "Transition", "Trim", "Hash Modification"]}
        selected_preset = None
        selected_methods: list[str] = []
        if effects.get("Hash Modification"):
            preset_map = {"fast": "⚡ Fast", "normal": "⚖️ Balanced", "slow": "🎯 Maximum"}
            selected_preset = st.radio("Preset", list(preset_map.keys()), format_func=lambda k: preset_map[k], index=1)
            st.info(HASH_PRESETS[selected_preset]["description"])
            selected_methods = st.multiselect(
                "Methods",
                options=HASH_PRESETS[selected_preset]["methods"],
                default=HASH_PRESETS[selected_preset]["methods"]
            )
        st.subheader("Processing Options")
        sample_rate = st.slider("Frame Sample Rate", 1, 60, 30)
        quality = st.select_slider("Output Quality", options=["Low", "Medium", "High"], value="Medium")
        
        # Add advanced settings
        add_advanced_settings(st.sidebar)

    # Main panel
    if uploaded_file:
        # Preview
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(uploaded_file.getvalue())
            vp = tmp.name
        # Don't add preview to temp_files yet!
        
        st.video(vp)
        
        # Process button
        if st.button("Process Video"):
            try:
                with st.spinner("Processing video variations..."):
                    props, variations, input_tmp_path = process_video(
                        uploaded_file, num_variations, effects,
                        selected_preset, selected_methods,
                        quality, sample_rate
                    )
                
                # Filter out any failed variations
                successful_variations = [var for var in variations if var['path'] is not None]
                
                if successful_variations:
                    # Calculate hash before adding to temp_files
                    orig_hash = calculate_video_hash(vp)['hash']
                    st.info(f"Original Hash: {orig_hash}")
                    
                    st.info(
                        f"Duration: {props['duration']:.2f}s, Resolution: {props['width']}x{props['height']}, FPS: {props['fps']}"
                    )
                    
                    # Show number of successful variations
                    if len(successful_variations) < num_variations:
                        st.warning(f"Completed {len(successful_variations)} out of {num_variations} requested variations. Some variations failed to process.")
                    
                    for idx, var in enumerate(successful_variations, start=1):
                        st.subheader(f"Variation {idx}")
                        st.write("Effects: " + ", ".join(var['effects']))
                        
                        # Calculate variation hash
                        vh = calculate_video_hash(var['path'])['hash']
                        diff = calculate_video_difference(vp, var['path'])
                        st.write(f"Variation Hash: {vh} (Diff: {diff:.2f}%)")
                        
                        st.video(var['path'])
                    
                    # Now create a ZIP with all variations
                    # Ensure output directory exists
                    os.makedirs("output", exist_ok=True)
                    
                    zipf = f"output/variations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
                    with zipfile.ZipFile(zipf, 'w') as zf:
                        for var in successful_variations:
                            if var['path'] and os.path.exists(var['path']):
                                zf.write(var['path'], os.path.basename(var['path']))
                    
                    with open(zipf, 'rb') as f:
                        st.download_button("Download All Variations", f, os.path.basename(zipf), mime='application/zip')
                    
                    # NOW add temp files for cleanup AFTER all processing is done
                    temp_files.append(vp)
                    temp_files.append(input_tmp_path)
                else:
                    st.error("No successful variations were produced. Please try again with different settings.")
                    # Clean up files if no successful variations
                    temp_files.append(vp)
                    temp_files.append(input_tmp_path)
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                logger.error(f"Processing error: {str(e)}\n{traceback.format_exc()}")
                # Clean up files on error
                if 'vp' in locals():
                    temp_files.append(vp)
                if 'input_tmp_path' in locals():
                    temp_files.append(input_tmp_path)
        else:
            # Add to temp_files only if process button wasn't clicked
            temp_files.append(vp)
    else:
        st.info("👆 Upload a video to get started!")

    st.markdown("---")
    st.markdown("<div style='text-align:center;'>Made with ❤️ using Streamlit</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()