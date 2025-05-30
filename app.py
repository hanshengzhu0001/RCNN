from flask import Flask, render_template, request, jsonify, make_response
from flask_cors import CORS
import os
import whisper
import json
from datetime import datetime
import torch
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import uuid
from pydub import AudioSegment
from ultralytics import YOLO
from skimage import measure
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
import io
import colorsys
from skimage.measure import label, find_contours, approximate_polygon
import traceback
from segment_anything import sam_model_registry, SamPredictor
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# State file for processed images per user
PROCESSED_IMAGES_FILE = "processed_images.json"

# Initialize Whisper model
whisper_model = whisper.load_model("base")

# Initialize Mask R-CNN
weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
mask_rcnn = maskrcnn_resnet50_fpn_v2(weights=weights)
mask_rcnn.eval()  # Set to evaluation mode
transform = weights.transforms()

# Initialize SAM
sam_checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
sam_predictor = SamPredictor(sam)

# COCO class mapping
COCO_CLASSES = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck',
    9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
    24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
    34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
    40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
    46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana',
    53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza',
    60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
    70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone',
    78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
    86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'
}

print("Mask R-CNN model loaded successfully")
print("SAM model loaded successfully")

# Initialize YOLOv10 model for object detection
yolo_model = YOLO('yolov10n.pt')  # Using the nano model for speed
print("YOLOv10 model loaded successfully")

# Create necessary directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("transcriptions", exist_ok=True)
os.makedirs("static/images", exist_ok=True)

# Sample images
SAMPLE_IMAGES = [
    "object1.jpg",     # New object image
    "landscape1.jpg",  # Original landscape images
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
        response = make_response(jsonify({"image": SAMPLE_IMAGES[0]}))
        response.set_cookie('user_id', user_id)

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

        if next_image is None:
            processed_data[user_id] = []
            save_processed_images(processed_data)
            next_image = SAMPLE_IMAGES[0]
            
        response = make_response(jsonify({"image": next_image}))

    return response

@app.route('/api/pre-segment', methods=['POST'])
def pre_segment():
    try:
        data = request.json
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        image_array = np.array(image)
        
        # Get image dimensions
        height, width = image_array.shape[:2]
        print(f"Processing image of size: {width}x{height}")
        
        # Transform image for Mask R-CNN
        img_tensor = transform(image)
        
        # Run Mask R-CNN prediction
        with torch.no_grad():
            prediction = mask_rcnn([img_tensor])[0]
        
        # Get unique regions
        unique_regions = []
        used_pixels = np.zeros((height, width), dtype=bool)
        
        # First pass: create background regions
        # Find connected components in the entire image
        labeled_image = label(np.ones((height, width), dtype=bool))
        for region_id in range(1, labeled_image.max() + 1):
            mask = labeled_image == region_id
            
            # Skip if region is too small
            if np.sum(mask) < 100:  # Minimum 100 pixels
                continue
            
            # Get contours for visualization
            contours = find_contours(mask, 0.5)
            polygons = []
            for contour in contours:
                # Simplify polygon
                contour = approximate_polygon(contour, tolerance=1.0)
                # Ensure polygon is closed
                if len(contour) > 0 and not np.array_equal(contour[0], contour[-1]):
                    contour = np.vstack((contour, contour[0]))
                # Fix diagonal reflection by swapping x and y coordinates
                contour = np.flip(contour, axis=1)
                polygons.append(contour.tolist())
            
            # Generate a unique color for this background region
            color = [int(c * 255) for c in colorsys.hsv_to_rgb(0.5, 0.3, 0.8)]  # Grayish color
            
            # Add background region to list
            unique_regions.append({
                'id': len(unique_regions),
                'class': 'background',
                'score': 1.0,
                'polygons': polygons,
                'color': color
            })
        
        # Second pass: overlay detected objects
        for idx in range(len(prediction['masks'])):
            mask = prediction['masks'][idx][0].numpy() > 0.5
            score = float(prediction['scores'][idx])
            class_id = int(prediction['labels'][idx])
            
            # Skip if score is too low
            if score < 0.5:  # Confidence threshold
                continue
                
            # Skip if mask is too small
            if np.sum(mask) < 100:  # Minimum 100 pixels
                continue
            
            # Get class name
            class_name = COCO_CLASSES.get(class_id, f"unknown_{class_id}")
            
            # Get contours for visualization
            contours = find_contours(mask, 0.5)
            polygons = []
            for contour in contours:
                # Simplify polygon
                contour = approximate_polygon(contour, tolerance=1.0)
                # Ensure polygon is closed
                if len(contour) > 0 and not np.array_equal(contour[0], contour[-1]):
                    contour = np.vstack((contour, contour[0]))
                # Fix diagonal reflection by swapping x and y coordinates
                contour = np.flip(contour, axis=1)
                polygons.append(contour.tolist())
            
            # Generate a unique color for this region
            color = [int(c * 255) for c in colorsys.hsv_to_rgb(class_id / 90, 0.8, 0.8)]
            
            # Add region to list
            unique_regions.append({
                'id': len(unique_regions),
                'class': class_name,
                'score': score,
                'polygons': polygons,
                'color': color
            })
            
            # Mark pixels as used
            used_pixels = np.logical_or(used_pixels, mask)
        
        print(f"Generated {len(unique_regions)} unique regions")
        return jsonify({'regions': unique_regions})
        
    except Exception as e:
        print(f"Error in pre-segment: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/segment', methods=['POST'])
def segment():
    try:
        data = request.json
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        image_array = np.array(image)
        
        # Get point coordinates
        x = int(data['x'])
        y = int(data['y'])
        
        # Check if point is in a background region
        is_background = True
        for region in data.get('regions', []):
            if region['class'] != 'background':
                # Check if point is inside any non-background region
                for polygon in region['polygons']:
                    if point_in_polygon(x, y, polygon):
                        is_background = False
                        break
                if not is_background:
                    break
        
        if not is_background:
            return jsonify({'error': 'Point is not in a background region'}), 400
        
        # Set image in SAM predictor
        sam_predictor.set_image(image_array)
        
        # Get mask from SAM
        masks, scores, _ = sam_predictor.predict(
            point_coords=np.array([[x, y]]),
            point_labels=np.array([1]),
            multimask_output=True
        )
        
        # Get the best mask
        best_mask = masks[np.argmax(scores)]
        
        # Get contours for visualization
        contours = find_contours(best_mask, 0.5)
        polygons = []
        for contour in contours:
            # Simplify polygon
            contour = approximate_polygon(contour, tolerance=1.0)
            # Ensure polygon is closed
            if len(contour) > 0 and not np.array_equal(contour[0], contour[-1]):
                contour = np.vstack((contour, contour[0]))
            # Fix diagonal reflection by swapping x and y coordinates
            contour = np.flip(contour, axis=1)
            polygons.append(contour.tolist())
        
        # Generate a unique color for this region
        color = [int(c * 255) for c in colorsys.hsv_to_rgb(0.5, 0.3, 0.8)]  # Grayish color
        
        return jsonify({
            'id': -1,  # New region
            'class': 'background',
            'score': float(np.max(scores)),
            'polygons': polygons,
            'color': color
        })
        
    except Exception as e:
        print(f"Error in segment: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def point_in_polygon(x, y, polygon):
    """Check if a point is inside a polygon using ray casting algorithm."""
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

@app.route('/api/detect', methods=['POST'])
def detect_objects():
    print("Received detection request")
    data = request.json
    image_data = data.get('image')
    
    try:
        # Convert base64 image to numpy array
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        image_height, image_width = np.array(image).shape[:2]
        
        # Transform image for Mask R-CNN
        img_tensor = transform(image)
        
        # Run Mask R-CNN prediction
        with torch.no_grad():
            prediction = mask_rcnn([img_tensor])[0]
        
        # Process results
        detections = []
        for idx in range(len(prediction['boxes'])):
            box = prediction['boxes'][idx].numpy()
            score = float(prediction['scores'][idx])
            class_id = int(prediction['labels'][idx])
            
            if score > 0.5:  # Confidence threshold
                class_name = COCO_CLASSES.get(class_id, f"unknown_{class_id}")
                
                detections.append({
                    "box": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                    "confidence": score,
                    "class": class_name
                })
        
        return jsonify({
            "detections": detections
        })

    except Exception as e:
        print(f"Error during detection: {e}")
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

def refine_transcription_with_gpt(transcription, image_filename):
    try:
        prompt = f"""As an AI trained to generate detailed image descriptions, please refine and expand the following transcription to make it more suitable for training visual language models. The transcription is about the image named '{image_filename}'.

Original transcription:
{transcription}

Please provide a refined version that:
1. Is more detailed and descriptive
2. Uses precise and specific language
3. Captures spatial relationships between objects
4. Includes relevant attributes (colors, sizes, textures)
5. Maintains a natural, flowing narrative
6. Focuses on visual elements that would be valuable for training vision-language models

Refined description:"""

        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are an expert at generating detailed, high-quality image descriptions for training vision-language models."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in GPT refinement: {str(e)}")
        return None

@app.route('/api/transcribe', methods=['POST'])
def transcribe():
    user_id = request.cookies.get('user_id')
    if not user_id:
        return jsonify({"error": "User ID not found"}), 400

    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_path = f"uploads/recording_{timestamp}.wav"
    audio_file.save(audio_path)
    
    # Get image filename from form data
    image_filename = request.form.get('image_filename')
    
    # Load the full audio recording
    full_audio = AudioSegment.from_file(audio_path)
    
    # Get click timestamps from form data (in ms)
    click_timestamps_str = request.form.get('click_timestamps')
    click_timestamps = json.loads(click_timestamps_str) if click_timestamps_str else []
    
    # Define fixed duration to save before the first click (in ms)
    PRE_CLICK_DURATION_MS = 2000  # 2 seconds

    segmented_transcriptions = []
    
    # Determine segment start and end times
    segment_times = []  # List of (start_time_ms, end_time_ms)
    
    if not click_timestamps:
        segment_times.append((0, len(full_audio)))
    else:
        first_click_time = click_timestamps[0]
        start_time = max(0, first_click_time - PRE_CLICK_DURATION_MS)
        end_time = first_click_time
        segment_times.append((start_time, end_time))

        for i in range(len(click_timestamps) - 1):
            start_time = click_timestamps[i]
            end_time = click_timestamps[i+1]
            segment_times.append((start_time, end_time))

        last_click_time = click_timestamps[-1]
        start_time = last_click_time
        end_time = len(full_audio)
        segment_times.append((start_time, end_time))

    # Process each segment
    for i, (start_time, end_time) in enumerate(segment_times):
        if start_time >= end_time:
            continue  # Skip empty segments
            
        segment = full_audio[start_time:end_time]
        segment_path = f"uploads/segment_{timestamp}_{i}.wav"
        segment.export(segment_path, format="wav")
        
        result = whisper_model.transcribe(segment_path)
        segment_transcription = result["text"]
        
        segmented_transcriptions.append({
            "segment_index": i,
            "start_time_ms": start_time,
            "end_time_ms": end_time,
            "transcription": segment_transcription.strip()
        })
        
        os.remove(segment_path)
    
    # Save segmented transcriptions
    segmented_transcription_path = f"transcriptions/segmented_transcription_{timestamp}.json"
    with open(segmented_transcription_path, "w") as f:
        json.dump(segmented_transcriptions, f, indent=4)
    
    # Combine all transcriptions for GPT refinement
    combined_transcription = "\n".join([seg["transcription"] for seg in segmented_transcriptions])
    
    # Get refined transcription from GPT
    refined_transcription = refine_transcription_with_gpt(combined_transcription, image_filename)
    
    # Mark image as processed for the user
    image_filename = request.form.get('image_filename')
    if image_filename:
        processed_data = load_processed_images()
        if user_id not in processed_data:
            processed_data[user_id] = []
        if image_filename not in processed_data[user_id]:
            processed_data[user_id].append(image_filename)
        save_processed_images(processed_data)
    
    return jsonify({
        "segmented_transcriptions": segmented_transcriptions,
        "refined_transcription": refined_transcription,
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

@app.route('/api/save-objects', methods=['POST'])
def save_objects():
    data = request.json
    image_filename = data.get('image_filename')
    objects = data.get('objects', [])
    
    if not image_filename:
        return jsonify({"error": "Missing image filename"}), 400
    
    # Create a unique filename for the objects data
    base_filename = os.path.splitext(image_filename)[0]
    objects_path = f"static/objects/{base_filename}_objects.json"
    
    # Create objects directory if it doesn't exist
    os.makedirs("static/objects", exist_ok=True)
    
    # Save objects data
    with open(objects_path, 'w') as f:
        json.dump({
            "image_filename": image_filename,
            "objects": objects,
            "timestamp": datetime.now().isoformat()
        }, f, indent=4)
    
    return jsonify({"success": True})

@app.route('/api/get-objects/<image_filename>')
def get_objects(image_filename):
    base_filename = os.path.splitext(image_filename)[0]
    objects_path = f"static/objects/{base_filename}_objects.json"
    
    if os.path.exists(objects_path):
        with open(objects_path, 'r') as f:
            return jsonify(json.load(f))
    
    return jsonify({"objects": []})

if __name__ == '__main__':
    app.run(debug=True) 