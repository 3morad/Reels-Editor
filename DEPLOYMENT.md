# Reels Editor Deployment Guide

## Option 1: Local Installation

### Prerequisites
- Python 3.9 or higher
- FFmpeg installed and in PATH
- Visual C++ Build Tools (Windows only)
- Git (optional)

### Steps
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd reels-editor
   ```

2. Create and activate virtual environment:
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   python main.py input/your_video.mp4 --voiceover
   ```

## Option 2: Docker Deployment

### Prerequisites
- Docker installed
- Git (optional)

### Steps
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd reels-editor
   ```

2. Build the Docker image:
   ```bash
   docker build -t reels-editor .
   ```

3. Run the container:
   ```bash
   # Mount input and output directories
   docker run -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output reels-editor input/your_video.mp4 --voiceover
   ```

## Option 3: Cloud Deployment (AWS Lambda)

### Prerequisites
- AWS account
- AWS CLI configured
- Docker installed

### Steps
1. Build the Lambda container:
   ```bash
   docker build -t reels-editor-lambda -f Dockerfile.lambda .
   ```

2. Push to Amazon ECR:
   ```bash
   aws ecr create-repository --repository-name reels-editor
   docker tag reels-editor-lambda:latest <account-id>.dkr.ecr.<region>.amazonaws.com/reels-editor:latest
   docker push <account-id>.dkr.ecr.<region>.amazonaws.com/reels-editor:latest
   ```

3. Create Lambda function using the container image

## Common Issues and Solutions

1. **Visual C++ Build Tools Error**
   - Windows: Install Visual C++ Build Tools from Microsoft
   - Linux: Install build-essential package
   - Docker: Already included in the container

2. **FFmpeg Not Found**
   - Windows: Add FFmpeg to PATH
   - Linux: Install FFmpeg package
   - Docker: Already included in the container

3. **Memory Issues**
   - Increase system swap space
   - Use smaller video files
   - Adjust processing parameters

4. **GPU Acceleration**
   - Install CUDA toolkit
   - Use NVIDIA Docker runtime
   - Set CUDA_VISIBLE_DEVICES environment variable

## Performance Optimization

1. **Batch Processing**
   ```bash
   # Process multiple videos
   python main.py --batch
   ```

2. **Resource Management**
   - Adjust number of variations
   - Use lower resolution videos
   - Enable GPU acceleration

3. **Storage Management**
   - Regular cleanup of output directory
   - Use external storage for large files
   - Implement file rotation

## Security Considerations

1. **Input Validation**
   - Validate video formats
   - Check file sizes
   - Scan for malware

2. **Resource Limits**
   - Set maximum file size
   - Limit concurrent processes
   - Implement rate limiting

3. **Access Control**
   - Use environment variables for sensitive data
   - Implement user authentication
   - Set up proper file permissions 