# Image Voiceover and Segmentation Platform

This web application allows users to:
- View images and record voiceovers
- Automatically transcribe voiceovers
- Edit refined transcriptions
- Use Meta's Segment Anything Model for image segmentation

## Setup

1. Install system dependencies:
```bash
# On macOS:
brew install ffmpeg

# On Ubuntu/Debian:
sudo apt-get install ffmpeg

# On Windows:
# Download and install ffmpeg from https://ffmpeg.org/download.html
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```

5. Open your browser and navigate to `http://localhost:5000`

## Features
- Image display and voiceover recording
- Automatic transcription using Whisper
- Transcription refinement
- Image segmentation using Meta's SAM model
- Modern, responsive UI

## Project Structure
- `app.py`: Flask backend
- `static/`: Static files (images, CSS, JS)
- `templates/`: HTML templates
- `uploads/`: Directory for uploaded files
- `transcriptions/`: Directory for stored transcriptions 