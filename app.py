from flask import Flask, render_template, request, jsonify, make_response
from flask_cors import CORS
import os
import whisper
import json
from datetime import datetime
from segment_anything import sam_model_registry, SamPredictor
import torch
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import uuid

app = Flask(__name__)
CORS(app)

# State file for processed images per user
PROCESSED_IMAGES_FILE = "processed_images.json"

# Initialize Whisper model
whisper_model = whisper.load_model("base")

# Initialize SAM model
sam_checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# Create necessary directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("transcriptions", exist_ok=True)
os.makedirs("static/images", exist_ok=True)

# Sample images (replace with your actual images)
SAMPLE_IMAGES = [
    "landscape1.jpg",
    "landscape2.jpg",
    "landscape3.jpg"
]

def load_processed_images():
    if os.path.exists(PROCESSED_IMAGES_FILE):
        with open(PROCESSED_IMAGES_FILE, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def save_processed_images(data):
    with open(PROCESSED_IMAGES_FILE, 'w') as f:
        json.dump(data, f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/get-image')
def get_image():
    user_id = request.cookies.get('user_id')
    if not user_id:
        user_id = str(uuid.uuid4())
        # Create a response using jsonify first
        response = make_response(jsonify({"image": SAMPLE_IMAGES[0]}))
        response.set_cookie('user_id', user_id)

        # Initialize processed images for the new user
        processed_data = load_processed_images()
        processed_data[user_id] = []
        save_processed_images(processed_data)

    else:
        processed_data = load_processed_images()
        processed_images = processed_data.get(user_id, [])

        next_image = None
        for image in SAMPLE_IMAGES:
            if image not in processed_images:
                next_image = image
                break

        # If all images processed, reset and get the first image
        if next_image is None:
            processed_data[user_id] = []
            save_processed_images(processed_data)
            next_image = SAMPLE_IMAGES[0]
            
        # Create a response using jsonify
        response = make_response(jsonify({"image": next_image}))

    return response

@app.route('/api/transcribe', methods=['POST'])
def transcribe():
    user_id = request.cookies.get('user_id')
    if not user_id:
         return jsonify({"error": "User ID not found"}), 400 # Should not happen if /get-image is called first

    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_path = f"uploads/recording_{timestamp}.wav"
    audio_file.save(audio_path)
    
    # Transcribe using Whisper
    result = whisper_model.transcribe(audio_path)
    transcription = result["text"]
    
    # Save transcription
    transcription_path = f"transcriptions/transcription_{timestamp}.txt"
    with open(transcription_path, "w") as f:
        f.write(transcription)
        
    # Mark image as processed for the user
    image_filename = request.form.get('image_filename') # Get filename from form data
    if image_filename:
        processed_data = load_processed_images()
        if user_id not in processed_data:
            processed_data[user_id] = []
        if image_filename not in processed_data[user_id]:
            processed_data[user_id].append(image_filename)
        save_processed_images(processed_data)
    
    return jsonify({
        "transcription": transcription,
        "timestamp": timestamp
    })

@app.route('/api/save-refined', methods=['POST'])
def save_refined():
    data = request.json
    timestamp = data.get('timestamp')
    refined_text = data.get('refined_text')
    
    if not timestamp or not refined_text:
        return jsonify({"error": "Missing required fields"}), 400
    
    refined_path = f"transcriptions/refined_{timestamp}.txt"
    with open(refined_path, "w") as f:
        f.write(refined_text)
    
    return jsonify({"success": True})

@app.route('/api/segment', methods=['POST'])
def segment():
    print("Received segmentation request") # Debug print
    data = request.json
    image_data = data.get('image')
    points = data.get('points')
    
    print(f"Image data received (first 50 chars): {image_data[:50]}...") # Debug print
    print(f"Points received: {points}") # Debug print

    try:
        # Convert base64 image to numpy array
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        image_array = np.array(image)
        
        # Set image in predictor
        predictor.set_image(image_array)
        
        # Convert points to input format
        input_points = np.array(points)
        input_labels = np.ones(len(points))
        
        print("Attempting SAM prediction...") # Debug print before prediction
        # Generate mask
        masks, scores, logits = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True
        )
        
        print(f"Segmentation masks generated: {len(masks)}") # Debug print
        print(f"Segmentation scores: {scores}") # Debug print

        # Convert mask to base64
        mask_image = Image.fromarray(masks[0].astype(np.uint8) * 255)
        buffered = BytesIO()
        mask_image.save(buffered, format="PNG")
        mask_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        response_data = {
            "mask": f"data:image/png;base64,{mask_base64}",
            "score": float(scores[0])
        }
        print("Sending segmentation response") # Debug print
        return jsonify(response_data)

    except Exception as e:
        print(f"Error during segmentation: {e}") # Log the error
        return jsonify({"error": str(e)}), 500 # Return error to frontend

if __name__ == '__main__':
    app.run(debug=True) 