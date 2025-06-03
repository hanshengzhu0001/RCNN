import os
import cv2
import numpy as np
import easyocr
import json
from typing import List, Dict, Tuple, Optional
import logging
from inference_sdk import InferenceHTTPClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScientificImageAnalyzer:
    def __init__(self):
        self.client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key="5sdQcX2eR75Oi87clCHI"
        )
        self.model_id = "chart-images-500/2"
        self.reader = easyocr.Reader(['en'])
        self.detection_threshold = 0.3
        self.ocr_threshold = 0.5
        self.chart_classes = [
            'bar', 'line', 'dot-line', 'legend', 'title',
            'x-axis-tick', 'x-axis-label', 'y-axis-tick', 'y-axis-label'
        ]

    def load_image(self, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def detect_chart_elements(self, image_path: str) -> List[Dict]:
        try:
            result = self.client.infer(image_path, model_id=self.model_id)
            detections = []
            for pred in result.get("predictions", []):
                if pred["confidence"] > self.detection_threshold:
                    x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
                    x1, y1 = x - w / 2, y - h / 2
                    x2, y2 = x + w / 2, y + h / 2
                    detections.append({
                        "box": [float(x1), float(y1), float(x2), float(y2)],
                        "class": pred["class"],
                        "confidence": float(pred["confidence"])
                    })
            return detections
        except Exception as e:
            logger.error(f"Error in chart element detection: {e}")
            return []

    def extract_text(self, image: np.ndarray) -> List[Dict]:
        try:
            results = self.reader.readtext(image)
            text_detections = []
            for (bbox, text, prob) in results:
                if prob > self.ocr_threshold:
                    x1, y1 = bbox[0]
                    x2, y2 = bbox[2]
                    text_detections.append({
                        'box': [float(x1), float(y1), float(x2), float(y2)],
                        'text': text,
                        'confidence': float(prob)
                    })
            return text_detections
        except Exception as e:
            logger.error(f"Error in text extraction: {e}")
            return []

    def filter_tiny_text(self, detections: List[Dict], min_area: int = 100) -> List[Dict]:
        filtered = []
        for det in detections:
            x1, y1, x2, y2 = det['box']
            area = (x2 - x1) * (y2 - y1)
            if area >= min_area:
                filtered.append(det)
        return filtered

    def resolve_overlapping_boxes(self, chart_detections: List[Dict], text_detections: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        def calculate_iou(box1, box2):
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = box1_area + box2_area - intersection
            return intersection / union if union > 0 else 0

        filtered_chart = []
        text_indices_to_remove = set()

        # Process chart detections first
        for i, chart_det in enumerate(chart_detections):
            keep_chart = True
            overlapping_text_indices = []

            # Find overlapping text detections
            for j, text_det in enumerate(text_detections):
                iou = calculate_iou(chart_det['box'], text_det['box'])
                if iou > 0.5: # Significant overlap
                    overlapping_text_indices.append(j)

            if chart_det['confidence'] >= 0.5:
                # Prioritize chart detections with confidence >= 0.5
                filtered_chart.append(chart_det)
                # Mark overlapping text detections for removal
                text_indices_to_remove.update(overlapping_text_indices)
            else:
                # For chart detections with confidence < 0.5, use confidence-based resolution
                for j in overlapping_text_indices:
                    text_det = text_detections[j]
                    if text_det['confidence'] > chart_det['confidence']:
                        # If overlapping text has higher confidence, don't keep this chart detection
                        keep_chart = False
                        break
                if keep_chart:
                    filtered_chart.append(chart_det)

        # Filter text detections: keep only those not marked for removal and not significantly overlapping with kept charts
        filtered_text = []
        kept_chart_boxes = [det['box'] for det in filtered_chart]

        for j, text_det in enumerate(text_detections):
            if j not in text_indices_to_remove:
                is_overlapping_with_kept_chart = False
                for kept_chart_box in kept_chart_boxes:
                    iou = calculate_iou(text_det['box'], kept_chart_box)
                    if iou > 0.5:
                        is_overlapping_with_kept_chart = True
                        break
                if not is_overlapping_with_kept_chart:
                    filtered_text.append(text_det)

        return filtered_chart, filtered_text

    def generate_caption(self, chart_detections: List[Dict], text_detections: List[Dict]) -> str:
        caption_parts = []
        all_detections = []
        for det in chart_detections:
            all_detections.append({
                'type': 'chart',
                'data': det,
                'priority': self.chart_classes.index(det['class']) if det['class'] in self.chart_classes else len(self.chart_classes)
            })
        for det in text_detections:
            all_detections.append({
                'type': 'text',
                'data': det,
                'priority': 0
            })
        all_detections.sort(key=lambda x: x['priority'])
        for det in all_detections:
            if det['type'] == 'chart':
                chart_data = det['data']
                caption_parts.append(f"{chart_data['class']} (confidence: {chart_data['confidence']:.2f})")
            else:
                text_data = det['data']
                caption_parts.append(f"Text: {text_data['text']} (confidence: {text_data['confidence']:.2f})")
        return " | ".join(caption_parts)

    def analyze_image(self, image_path: str) -> Dict:
        try:
            image = self.load_image(image_path)
            chart_detections = self.detect_chart_elements(image_path)
            text_detections = self.extract_text(image)
            text_detections = self.filter_tiny_text(text_detections)
            filtered_chart, filtered_text = self.resolve_overlapping_boxes(chart_detections, text_detections)
            caption = self.generate_caption(filtered_chart, filtered_text)
            return {
                'chart_detections': filtered_chart,
                'text_detections': filtered_text,
                'caption': caption
            }
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return {
                'chart_detections': [],
                'text_detections': [],
                'caption': f"Error analyzing image: {str(e)}"
            }

if __name__ == "__main__":
    analyzer = ScientificImageAnalyzer()
    result = analyzer.analyze_image("static/images/science1.jpg")
    print(json.dumps(result, indent=2)) 