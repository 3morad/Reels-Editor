# Reels Editor - GPU Accelerated Video Processing

A high-performance video processing application for creating social media reels with AI-powered effects. This version includes GPU acceleration for significantly faster processing.

## Features

- **GPU Acceleration**: Process videos up to 50x faster with NVIDIA GPU support
- **Smart Resource Management**: Automatically adjusts batch sizes and processing parameters based on available resources
- **Multiple Effect Types**: Zoom, crop, filters, transitions, and more
- **Hash Modification**: Special effects to modify video hashes for unique content
- **Parallel Processing**: Process multiple variations simultaneously
- **Hardware-Accelerated Encoding**: Use NVENC for faster video exports

## Requirements

- Python 3.8+
- NVIDIA GPU with CUDA support (optional, but recommended)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/reels-editor.git
cd reels-editor
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. For GPU support (NVIDIA only):
   - Install CUDA Toolkit 11.x or 12.x from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads)
   - Uncomment and install the optional GPU dependencies in requirements.txt:
```bash
pip install cupy-cuda11x  # Choose version matching your CUDA installation
pip install pycuda
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open the application in your browser (usually http://localhost:8501)

3. Upload a video file

4. Configure settings:
   - Number of variations
   - Quality settings
   - Effects to apply
   - Advanced GPU settings (if available)

5. Click "Process Video" to start processing

6. Download results individually or as a ZIP file

## Advanced Configuration

### GPU Settings

- **Use GPU Acceleration**: Enable or disable GPU processing
- **GPU Memory Threshold**: Maximum GPU memory usage before falling back to CPU (0.1-0.95)
- **Batch Size**: Number of variations to process simultaneously
- **Max Workers**: Maximum number of parallel processes
- **Batch Cooldown**: Seconds to wait between batches to let system resources recover

### Effect Settings

- **Zoom**: Magnify the video
- **Crop**: Crop from edges and resize
- **Filter**: Apply color/brightness/contrast adjustments
- **Transition**: Add fade-in/fade-out effects
- **Trim**: Remove part of the end of the video
- **Hash Modification**: Apply special effects that modify video hash fingerprints

## Troubleshooting

### CUDA Issues

If you encounter CUDA errors:

1. Check if CUDA is properly installed:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

2. If you get "The library is compiled without CUDA support" error:
   - Reinstall PyTorch with CUDA support:
```bash
pip uninstall torch
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu117  # For CUDA 11.7
```

3. Check GPU memory usage with:
```bash
nvidia-smi
```

### Performance Optimization

- Reduce video resolution for faster processing
- Lower batch size if experiencing memory issues
- Reduce quality settings for faster processing
- Use fewer workers on systems with limited resources

## License

[MIT License](LICENSE)

## Acknowledgements

- [PyTorch](https://pytorch.org/) for GPU acceleration
- [MoviePy](https://zulko.github.io/moviepy/) for video processing
- [Streamlit](https://streamlit.io/) for the web interface
- [FFmpeg](https://ffmpeg.org/) for video encoding/decoding
