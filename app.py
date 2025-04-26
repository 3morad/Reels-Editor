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

# Try importing GPU libraries with graceful fallbacks
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from concurrent.futures import ThreadPoolExecutor 
from src.video.input import VideoInput
from src.video.core.transformer import VideoTransformer
from src.video.utils.logging_utils import configure_logger
from src.video.utils.hash_calculator import calculate_video_hash, calculate_video_difference
from src.video.effects.hash_presets import HASH_PRESETS, get_preset_default_intensity
from src.video.export import VideoExporter
from src.video.utils.gpu_manager import gpu_manager

# Configure logger
logger = configure_logger("App")
# Enable DEBUG level to see all debug logs
logger.setLevel(logging.DEBUG)

# Default configuration settings for batch processing
CONFIG = {
    "batch_size": 10,               # Number of variations to process at once
    "max_workers": 10,              # Maximum number of parallel processes
    "batch_cooldown_seconds": 2,   # Seconds to wait between batches
    "use_gpu": True,               # Whether to use GPU acceleration
    "gpu_memory_threshold": 0.8,   # GPU memory threshold (0.0-1.0)
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
    # Clean up GPU resources
    if TORCH_AVAILABLE and gpu_manager.has_cuda:
        gpu_manager.cleanup()
atexit.register(cleanup_temp_files)

# Top-level variation worker (must be pickle-friendly)
def _process_variation(args):
    tmp_path, i, effects, selected_preset, selected_methods, quality, fps = args
    try:
        start_time = time.time()
        logger.info(f"Starting variation {i+1} processing")
        
        # Set a unique random seed for each variation to ensure different randomization
        variation_seed = int(time.time() * 1000) + i * 1000 + random.randint(1, 999)
        random.seed(variation_seed)
        logger.info(f"Using random seed {variation_seed} for variation {i+1}")
        
        # Initialize video input
        vi = VideoInput(tmp_path)
        
        # Create transformer with GPU optimization if available
        tf = VideoTransformer(vi.video_clip, input_path=tmp_path)
        
        # Apply effects based on selection with randomized parameters
        if effects.get("Zoom"): 
            zoom_factor = random.uniform(1.01, 1.05)
            logger.debug(f"Variation {i+1}: zoom_factor={zoom_factor}")
            tf.apply_zoom(zoom_factor)
            
        if effects.get("Crop"): 
            crop_percent = random.uniform(0.01, 0.05)
            logger.debug(f"Variation {i+1}: crop_percent={crop_percent}")
            tf.apply_crop(crop_percent)
            
        if effects.get("Filter"): 
            filter_type = random.choice(["brightness", "contrast", "saturation"])
            intensity = random.uniform(0.1, 0.3)
            logger.debug(f"Variation {i+1}: filter={filter_type}, intensity={intensity}")
            tf.apply_filter(filter_type, intensity)
            
        if effects.get("Transition"): 
            trans_type = random.choice(["fadein", "fadeout"])
            duration = random.uniform(0.5, 1.5)
            logger.debug(f"Variation {i+1}: transition={trans_type}, duration={duration}")
            tf.apply_transition(trans_type, duration)
            
        if effects.get("Trim"): 
            trim_percent = random.uniform(0.1, 0.3)
            logger.debug(f"Variation {i+1}: trim_percent={trim_percent}")
            tf.apply_trim(trim_percent)
            
        if effects.get("Hash Modification"):
            # Apply each selected hash method with randomized intensity
            for ht in selected_methods:
                try:
                    # Get baseline intensity from preset
                    base = get_preset_default_intensity(selected_preset, ht)
                    
                    # Use wider random variation in intensity for each hash effect
                    # This ensures more unique results for each variation
                    intensity_variation = random.uniform(0.6, 1.4)  # 60% to 140% of base intensity
                    inten = max(0.1, min(1.0, base * intensity_variation))
                    
                    # Add a random minor offset for even more uniqueness
                    inten += random.uniform(-0.05, 0.05)
                    inten = max(0.1, min(1.0, inten))  # Ensure still within valid range
                    
                    logger.debug(f"Variation {i+1}: hash_type={ht}, intensity={inten}")
                    tf.modify_hash(ht, inten)
                except ValueError as e:
                    logger.warning(f"Hash method error: {e}. Using default intensity.")
                    # Use a randomized default intensity
                    inten = random.uniform(0.15, 0.25)
                    logger.debug(f"Variation {i+1}: hash_type={ht}, fallback_intensity={inten}")
                    tf.modify_hash(ht, inten)
        
        logger.info(f"Applied effects to variation {i+1}, transforming clip")
        
        # GPU optimization happens automatically in get_transformed_clip()
        # based on availability and memory resources
        clip = tf.get_transformed_clip()
        
        # Generate filename
        fname = f"variation_{i+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Map quality string to preset
        preset = 'ultrafast' if quality == 'Low' else 'medium' if quality == 'Medium' else 'slow'
        
        logger.info(f"Exporting variation {i+1} with preset {preset}")
        
        # Export with GPU acceleration if available
        out = exporter.export_video(
            video_clip=clip,
            filename=fname,
            fps=fps,
            export_settings={
                'preset': preset
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
        
        # Force garbage collection
        gc.collect()
        
        # Clean up GPU resources
        if TORCH_AVAILABLE and gpu_manager.has_cuda:
            gpu_manager.cleanup()
        
        processing_time = time.time() - start_time
        logger.info(f"Completed variation {i+1} in {processing_time:.2f} seconds")
            
        return out, eff_strs
    except Exception as e:
        logger.error(f"Error processing variation {i+1}: {str(e)}\n{traceback.format_exc()}")
        return None, [f"Error: {str(e)}"]

# Orchestrate parallel processing with batching
def process_video(uploaded_file, num_variations, effects, selected_preset, selected_methods, quality, sample_rate):
    st.session_state.processing_start_time = time.time()
    
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
    
    # GPU-aware batch sizing - smaller batches for GPU to avoid memory issues
    if CONFIG["use_gpu"] and gpu_manager.has_cuda:
        # Use smaller batches and fewer workers with GPU
        batch_size = min(10, CONFIG["batch_size"])
        max_workers = min(10,min(CONFIG["max_workers"], min(10, exporter.cpu_count)))
        logger.info(f"Using GPU-optimized batch settings: batch_size={batch_size}, workers={max_workers}")
    else:
        batch_size = CONFIG["batch_size"]
        max_workers = min(CONFIG["max_workers"], exporter.cpu_count)
        logger.info(f"Using CPU batch settings: batch_size={batch_size}, workers={max_workers}")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_batches = (len(tasks) + batch_size - 1) // batch_size
    
    # Display overall progress metrics
    start_time = time.time()
    metrics_placeholder = st.empty()
    
    for batch_idx in range(0, len(tasks), batch_size):
        batch = tasks[batch_idx:batch_idx + batch_size]
        current_batch = batch_idx // batch_size + 1
        
        status_text.text(f"Processing batch {current_batch}/{total_batches}...")
        logger.info(f"Starting batch {current_batch}/{total_batches} with {len(batch)} variations")
        
        batch_start_time = time.time()
        batch_results = []
        
        # Calculate and display metrics
        elapsed = time.time() - start_time
        variations_completed = len(results)
        variations_per_second = variations_completed / elapsed if elapsed > 0 else 0
        estimated_total = num_variations / variations_per_second if variations_per_second > 0 else 0
        remaining_time = max(0, estimated_total - elapsed)
        
        metrics_placeholder.info(
            f"Progress: {variations_completed}/{num_variations} variations | "
            f"Elapsed: {elapsed:.1f}s | "
            f"ETA: {remaining_time:.1f}s | "
            f"Speed: {variations_per_second:.2f} var/s"
        )
        
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for out, effs in pool.map(_process_variation, batch):
                if out is not None:  # Only add successful variations
                    batch_results.append({'path': out, 'effects': effs})
                
        results.extend(batch_results)
        
        # Update progress
        progress = min(1.0, (batch_idx + len(batch)) / len(tasks))
        progress_bar.progress(progress)
        
        batch_time = time.time() - batch_start_time
        logger.info(f"Completed batch {current_batch}/{total_batches} in {batch_time:.2f}s")
        
        # Force garbage collection between batches
        gc.collect()
        
        # Clean up GPU resources between batches
        if TORCH_AVAILABLE and gpu_manager.has_cuda:
            gpu_manager.cleanup()
        
        # Small delay to let system resources recover, shorter for smaller batches
        if batch_idx + batch_size < len(tasks):  # If not the last batch
            cooldown = max(1, CONFIG["batch_cooldown_seconds"] / 2) if batch_size <= 2 else CONFIG["batch_cooldown_seconds"]
            time.sleep(cooldown)
    
    total_time = time.time() - start_time
    logger.info(f"Completed all {len(results)} variations in {total_time:.2f}s")
    
    progress_bar.progress(1.0)
    status_text.text(f"Processing complete! Generated {len(results)} variations in {total_time:.2f}s")
    
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
        
        # GPU settings if available
        if TORCH_AVAILABLE and gpu_manager.has_cuda:
            st.caption("GPU Settings")
            CONFIG["use_gpu"] = st.checkbox("Use GPU Acceleration", 
                                          value=CONFIG["use_gpu"],
                                          help="Enable/disable GPU acceleration")
            CONFIG["gpu_memory_threshold"] = st.slider("GPU Memory Threshold", 
                                                     min_value=0.1, 
                                                     max_value=0.95, 
                                                     value=CONFIG["gpu_memory_threshold"],
                                                     step=0.05,
                                                     help="Maximum GPU memory usage before fallback to CPU")
            
            # Apply GPU settings
            gpu_manager.memory_threshold = CONFIG["gpu_memory_threshold"]
            gpu_manager.force_cpu = not CONFIG["use_gpu"]
            
            # Display GPU information
            if gpu_manager.has_cuda:
                st.info(f"GPU: {torch.cuda.get_device_name()}")
                memory_used = gpu_manager.get_memory_usage() * 100
                st.progress(memory_used / 100, f"GPU Memory: {memory_used:.1f}%")

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
        num_variations = st.slider("Number of Variations", 1, 65, 65)
        st.subheader("Select Effects to Apply")
        effects = {k: st.checkbox(k, value=False if k == 'Trim' else (k != 'Trim')) for k in ["Zoom", "Crop", "Filter", "Transition", "Trim", "Hash Modification"]}
        selected_preset = None
        selected_methods: list[str] = []
        if effects.get("Hash Modification"):
            # Make 'fast' preset the only option and selected by default
            selected_preset = "fast"
            st.info(HASH_PRESETS[selected_preset]["description"])
            selected_methods = st.multiselect(
                "Methods",
                options=HASH_PRESETS[selected_preset]["methods"],
                default=HASH_PRESETS[selected_preset]["methods"]
            )
        st.subheader("Processing Options")
        sample_rate = st.slider("Frame Sample Rate", 1, 60, 30)
        quality = st.select_slider("Output Quality", options=["Low", "Medium", "High"], value="Medium")
        
        # Add advanced settings with GPU controls
        add_advanced_settings(st.sidebar)
        
        # Show GPU status if available
        if TORCH_AVAILABLE and gpu_manager.has_cuda:
            st.sidebar.subheader("🚀 GPU Acceleration")
            st.sidebar.info(f"GPU: {torch.cuda.get_device_name()}")
            memory_used = gpu_manager.get_memory_usage() * 100
            st.sidebar.progress(memory_used / 100, f"Memory: {memory_used:.1f}%")
            if gpu_manager.has_nvenc:
                st.sidebar.success("Hardware encoding enabled ✓")

    # Main panel
    if uploaded_file:
        # Preview
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(uploaded_file.getvalue())
            vp = tmp.name
        # Don't add preview to temp_files yet!
        
        st.video(vp)
        
        # Process button
        process_col, download_col = st.columns(2)
        
        process_clicked = process_col.button("Process Video")
        
        # Initialize download button variable
        zipf = None
        
        if process_clicked:
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
                    # Skip hash calculation for better performance
                    
                    st.info(
                        f"Duration: {props['duration']:.2f}s, Resolution: {props['width']}x{props['height']}, FPS: {props['fps']}"
                    )
                    
                    # Show number of successful variations
                    if len(successful_variations) < num_variations:
                        st.warning(f"Completed {len(successful_variations)} out of {num_variations} requested variations. Some variations failed to process.")
                    
                    # Create a ZIP with all variations right away
                    # Ensure output directory exists
                    os.makedirs("output", exist_ok=True)
                    
                    zipf = f"output/variations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
                    with zipfile.ZipFile(zipf, 'w') as zf:
                        for var in successful_variations:
                            if var['path'] and os.path.exists(var['path']):
                                zf.write(var['path'], os.path.basename(var['path']))
                    
                    # Display videos in a grid layout
                    st.subheader("Generated Variations")
                    
                    # Calculate number of columns (3 is a good number for most screens)
                    num_cols = 3
                    rows = [successful_variations[i:i + num_cols] for i in range(0, len(successful_variations), num_cols)]
                    
                    for row_idx, row in enumerate(rows):
                        cols = st.columns(num_cols)
                        for col_idx, var in enumerate(row):
                            with cols[col_idx]:
                                st.write(f"**Variation {row_idx * num_cols + col_idx + 1}**")
                                st.write("Effects: " + ", ".join(var['effects']))
                                st.video(var['path'])
                    
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
        
        # Show download button if we have a ZIP file
        if st.session_state.get('zipf_path') or zipf:
            zip_path = zipf or st.session_state.get('zipf_path')
            if os.path.exists(zip_path):
                with open(zip_path, 'rb') as f:
                    download_col.download_button(
                        "Download All Variations", 
                        f, os.path.basename(zip_path), 
                        mime='application/zip'
                    )
                # Save the path for the next run
                st.session_state['zipf_path'] = zip_path
        
        # Add to temp_files only if process button wasn't clicked
        if not process_clicked:
            temp_files.append(vp)
    else:
        st.info("👆 Upload a video to get started!")
 
    st.markdown("---")
    st.markdown("<div style='text-align:center;'>Made with ❤️ using Streamlit</div>", unsafe_allow_html=True)
    
if __name__ == "__main__":
    main()