# Reels Editor

A Python-based video automation tool for creating multiple variations of videos with different effects, text detection, and AI voice-over capabilities.

## Features

- Video variation generation
- Text detection and relocation
- Multiple visual effects (zoom, crop, filters)
- Audio processing and effects
- AI voice-over generation
- Batch processing support

## Requirements

- Python 3.8+
- OpenCV
- MoviePy
- Tesseract OCR
- TTS (Text-to-Speech)

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

4. Install Tesseract OCR:
- Windows: Download and install from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
- Linux: `sudo apt-get install tesseract-ocr`
- macOS: `brew install tesseract`

## Usage

### Single Video Processing
```bash
python main.py --input input/video.mp4 --variations 50
```

### Batch Processing
```bash
python main.py --batch --variations 50
```

### With Voice-over
```bash
python main.py --input input/video.mp4 --voiceover
```

## Project Structure

```
reels-editor/
├── src/
│   ├── video/
│   │   ├── input.py
│   │   ├── transform.py
│   │   ├── text_processor.py
│   │   └── export.py
│   └── audio/
│       ├── processor.py
│       └── voice_over.py
├── input/
├── output/
├── config/
├── main.py
└── requirements.txt
```

## License

MIT License - see LICENSE file for details 