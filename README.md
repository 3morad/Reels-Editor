# Video Hash Modification Tool

A sophisticated tool for applying various modifications to videos while preserving visual quality, specifically designed to alter video hashes for content identification purposes. This tool implements multiple advanced techniques to modify video perceptual hashes while maintaining visual integrity.

## Features

### Hash Modification Techniques
- **DCT (Discrete Cosine Transform) Modifications**
  - Frequency domain alterations
  - High-frequency coefficient manipulation
  - Imperceptible visual changes

- **Temporal Pattern Modifications**
  - Frame pattern alterations
  - Strategic frame duplication
  - Temporal consistency preservation

- **Noise Addition**
  - PRNG-seeded noise patterns
  - Selective noise application
  - Edge-aware noise masking

- **Geometric Transformations**
  - Subtle perspective shifts
  - Non-linear warping
  - Micro-level spatial modifications

- **Color Space Manipulations**
  - LAB color space modifications
  - Chroma component adjustments
  - Color temperature variations

- **Pixel-Level Modifications**
  - Brightness variations
  - Gaussian noise addition
  - Controlled blur effects

- **Frame Delay/Insertion**
  - Strategic frame delays
  - Patterned frame insertion
  - Temporal consistency maintenance

- **Watermarking**
  - Pattern-based watermarks
  - Alpha-blended overlays
  - Position randomization

### Additional Video Effects
- Random zoom with configurable parameters
- Dynamic cropping with percentage control
- Multiple filter options (contrast, brightness, etc.)
- Smooth transitions between effects
- Advanced text detection and relocation

## Technical Implementation

### Core Components
- **Video Processing Pipeline**
  - Frame-by-frame processing
  - Real-time modification application
  - Quality preservation mechanisms

- **Hash Generation**
  - Perceptual hash calculation
  - Hash difference analysis
  - Modification effectiveness metrics

- **Effect Application**
  - Parallel processing capabilities
  - Memory-efficient implementation
  - Error handling and recovery

## Requirements

### System Requirements
- Python 3.8+
- FFmpeg for video processing
- Tesseract OCR for text detection
- 4GB+ RAM recommended
- GPU acceleration supported

### Python Dependencies
```
opencv-python>=4.8.1
moviepy>=1.0.3
Pillow>=10.0.0
streamlit>=1.28.0
numpy>=1.24.0
pytesseract>=0.3.10
videohash>=2.1.9
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/video-hash-modifier.git
cd video-hash-modifier
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

4. Install system dependencies:
- FFmpeg: [Installation guide](https://ffmpeg.org/download.html)
- Tesseract OCR: [Installation guide](https://github.com/tesseract-ocr/tesseract)

## Usage

1. Run the application:
```bash
streamlit run app.py
```

2. Web Interface Features:
   - Video upload and preview
   - Effect selection and configuration
   - Real-time processing status
   - Hash comparison visualization
   - Result download options

3. Processing Options:
   - Multiple hash modification methods
   - Custom effect parameters
   - Quality settings
   - Output format selection

## Configuration

### Default Settings
Modify `config.py` to adjust:
- Default effect parameters
- Processing thresholds
- Output quality settings
- Memory usage limits

### Advanced Customization
Edit `src/video/transform.py` to:
- Adjust modification intensities
- Add new effect types
- Modify processing algorithms
- Implement custom hash methods

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Include comprehensive docstrings
- Add unit tests for new features
- Update documentation accordingly

## License

MIT License

## Acknowledgments

- OpenCV for video processing capabilities
- MoviePy for video manipulation
- Streamlit for web interface
- Tesseract for OCR functionality 