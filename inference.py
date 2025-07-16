# inference.py

import os
import math
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import mmcv
from mmdet.structures import DetDataSample

from mmdet.utils import register_all_modules
from mmengine.config import Config
from mmengine.registry import MODELS
from mmdet.apis import inference_detector
from mmdet.registry import MODELS

# Import custom models
from custom_models.custom_cascade_with_meta import CustomCascadeWithMeta
from custom_models.custom_heads import FCHead, RegHead
from custom_models.register import register_custom_models
from custom_models.custom_dataset import ChartDataset
from custom_models.custom_transforms import PackChartInputs

# Register custom models (this just ensures the modules are imported)
register_custom_models()

# ─── 1) Monkey-patch torch.load so weights_only=False by default ───
_orig_torch_load = torch.load
def _torch_load_no_weights_only(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_no_weights_only
# ────────────────────────────────────────────────────────────────────

# ─── 2) Shim numpy._core.multiarray for NumPy ≥1.23 ───
import numpy__core_shim  # see previous instructions
# ────────────────────────────────────────────────────────────────────

def visualize_detections(img, instances, class_names, meta_info, score_thr=0.3, out_path=None, pred_data_series=None):
    """
    img: H×W×3 RGB numpy array
    instances: a DetDataSample.pred_instances object
    meta_info: dictionary containing chart metadata
    pred_data_series: list of dicts with keys 'x' and 'y' (model predictions)
    """
    plt.figure(figsize=(12,8))
    plt.imshow(img)
    ax = plt.gca()

    # Extract raw arrays
    bboxes = instances.bboxes.cpu().numpy()       # shape (N,4)
    scores = instances.scores.cpu().numpy()       # shape (N,)
    labels = instances.labels.cpu().numpy()       # shape (N,)

    # Convert chart type tensor to label
    if isinstance(meta_info.get('chart_type'), torch.Tensor):
        chart_type_tensor = meta_info['chart_type']
        chart_type_idx = torch.argmax(chart_type_tensor).item()
        confidence = torch.max(chart_type_tensor).item()
        chart_types = ['line', 'scatter', 'dot', 'vertical_bar', 'horizontal_bar']  # Updated chart types
        chart_type = chart_types[chart_type_idx] if confidence > 0.5 else 'Unknown'
    else:
        chart_type = meta_info.get('chart_type', 'Unknown')

    # Add chart type and metadata as title
    plt.title(f'Chart Type: {chart_type}', fontsize=12, pad=20)

    # Draw plot bounding box if available
    if 'plot_bb' in meta_info:
        x1, y1, x2, y2 = meta_info['plot_bb']
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                           fill=False, edgecolor='blue', linewidth=2, linestyle='--')
        ax.add_patch(rect)
        # Add plot area label at the top-left corner
        ax.text(x1, y1-5, "Plot Area", color='blue', fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7))

    # Draw axes if available
    if 'axes_info' in meta_info:
        axes_info = meta_info['axes_info']
        
        # Draw x-axis
        if 'x-axis' in axes_info:
            x_axis = axes_info['x-axis']
            # Draw main line
            plt.plot([x_axis['min'], x_axis['max']], 
                    [y1, y1],  # Use plot area y1 for x-axis
                    'g-', linewidth=2, label='X-axis')
            # Add axis label at the center of the axis
            plt.text((x_axis['min'] + x_axis['max'])/2, y1-20, 
                    f"X-axis ({x_axis['values-type']})", 
                    color='green', fontsize=8, ha='center',
                    bbox=dict(facecolor='white', alpha=0.7))
            # Add origin label at the start of x-axis
            plt.text(x_axis['min']-10, y1+5, "Origin", 
                    color='green', fontsize=8, ha='right',
                    bbox=dict(facecolor='white', alpha=0.7))
        
        # Draw y-axis
        if 'y-axis' in axes_info:
            y_axis = axes_info['y-axis']
            # Draw main line
            plt.plot([x1, x1],  # Use plot area x1 for y-axis
                    [y_axis['min'], y_axis['max']], 
                    'g-', linewidth=2, label='Y-axis')
            # Add axis label at the center of the axis
            plt.text(x1-20, (y_axis['min'] + y_axis['max'])/2, 
                    f"Y-axis ({y_axis['values-type']})", 
                    color='green', fontsize=8, va='center', rotation=90,
                    bbox=dict(facecolor='white', alpha=0.7))
            # Add origin label at the start of y-axis
            plt.text(x1+5, y_axis['min']-10, "Origin", 
                    color='green', fontsize=8, va='top',
                    bbox=dict(facecolor='white', alpha=0.7))

        # Draw origin point if both axes are available
        if 'x-axis' in axes_info and 'y-axis' in axes_info:
            x_axis = axes_info['x-axis']
            y_axis = axes_info['y-axis']
            plt.plot(x_axis['min'], y_axis['min'], 'go', markersize=8, label='Origin')
            plt.text(x_axis['min']-5, y_axis['min']-5, "O", 
                    color='green', fontsize=10, ha='right', va='top',
                    bbox=dict(facecolor='white', alpha=0.7))

    # Draw ground truth data series if available (in red)
    if 'data_series' in meta_info:
        for series in meta_info['data_series']:
            if series['type'] == 'line':
                points = np.array(series['points'])
                plt.plot(points[:, 0], points[:, 1], 'r-', linewidth=2, label='GT Series')
                plt.scatter(points[:, 0], points[:, 1], color='red', s=30)
                # Add point coordinates
                for x, y in points:
                    plt.text(x+5, y+5, f'({x:.1f},{y:.1f})', 
                            color='red', fontsize=6,
                            bbox=dict(facecolor='white', alpha=0.7))

    # Draw predicted data series if available (in magenta)
    if pred_data_series is not None and len(pred_data_series) > 0:
        pred_points = np.array([[pt['x'], pt['y']] for pt in pred_data_series])
        plt.plot(pred_points[:, 0], pred_points[:, 1], 'm--', linewidth=2, label='Predicted Series')
        plt.scatter(pred_points[:, 0], pred_points[:, 1], color='magenta', s=30)
        for x, y in pred_points:
            plt.text(x+5, y-10, f'({x:.1f},{y:.1f})', 
                    color='magenta', fontsize=6,
                    bbox=dict(facecolor='white', alpha=0.7))

    # Create legend for detected elements
    legend_elements = []
    legend_labels = []

    for idx in range(len(bboxes)):
        x1,y1,x2,y2 = bboxes[idx].astype(int)
        score = scores[idx]
        label = labels[idx]
        if score < score_thr:
            continue

        # Add bounding box
        rect = plt.Rectangle((x1,y1), x2-x1, y2-y1,
                           fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        
        # Add label with score
        ax.text(x1, y1-5,
                f"{class_names[label]}:{score:.2f}",
                color='red', fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7))
        
        # Add to legend
        legend_elements.append(rect)
        legend_labels.append(f"{class_names[label]}")

    # Add legend
    if legend_elements:
        ax.legend(legend_elements, legend_labels, 
                 loc='upper right', bbox_to_anchor=(1.15, 1))

    plt.axis('off')
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

def print_detection_results(instances, class_names, meta_info, score_thr=0.3):
    """Print detailed detection results including metadata"""
    print("\n" + "="*50)
    print("DETECTION RESULTS")
    print("="*50)
    
    # Get image dimensions from meta_info
    img_shape = meta_info.get('img_shape', (800, 1333))  # Default to common size if not available
    img_height, img_width = img_shape[0], img_shape[1]
    print(f"\nImage Dimensions: {img_width}x{img_height}")
    
    # Print chart type with confidence if available
    if isinstance(meta_info.get('chart_type'), torch.Tensor):
        chart_type_tensor = meta_info['chart_type']
        confidence = torch.max(chart_type_tensor).item()
        chart_type_idx = torch.argmax(chart_type_tensor).item()
        chart_types = ['line', 'scatter', 'dot', 'vertical_bar', 'horizontal_bar']
        chart_type = chart_types[chart_type_idx] if confidence > 0.5 else 'Unknown'
        print(f"\nChart Type: {chart_type} (confidence: {confidence:.3f})")
    else:
        print(f"\nChart Type: {meta_info.get('chart_type', 'Unknown')}")
    
    # Print plot area if available
    if 'plot_bb' in meta_info:
        x1, y1, x2, y2 = meta_info['plot_bb']
        # Scale coordinates to image size (assuming plot_bb is in normalized coordinates)
        x1, x2 = int(x1 * img_width), int(x2 * img_width)
        y1, y2 = int(y1 * img_height), int(y2 * img_height)
        print("\nPlot Area:")
        print(f"  Position: [{x1}, {y1}, {x2}, {y2}]")
        print(f"  Width: {x2-x1}, Height: {y2-y1}")
    
    # Print axes information if available
    if 'axes_info' in meta_info:
        print("\nAxes Information:")
        axes_info = meta_info['axes_info']
        
        if 'x-axis' in axes_info:
            x_axis = axes_info['x-axis']
            # Scale x-axis coordinates (assuming normalized coordinates)
            x_min, x_max = int(x_axis['min'] * img_width), int(x_axis['max'] * img_width)
            print("  X-axis:")
            print(f"    Range: [{x_min}, {x_max}]")
            print(f"    Type: {x_axis['values-type']}")
            print(f"    Tick Type: {x_axis['tick-type']}")
        
        if 'y-axis' in axes_info:
            y_axis = axes_info['y-axis']
            # Scale y-axis coordinates (assuming normalized coordinates)
            y_min, y_max = int(y_axis['min'] * img_height), int(y_axis['max'] * img_height)
            print("  Y-axis:")
            print(f"    Range: [{y_min}, {y_max}]")
            print(f"    Type: {y_axis['values-type']}")
            print(f"    Tick Type: {y_axis['tick-type']}")
    
    # Print data series information if available
    if 'data_series' in meta_info:
        print("\nData Series Information:")
        for i, series in enumerate(meta_info['data_series']):
            print(f"  Series {i+1}:")
            print(f"    Type: {series['type']}")
            print(f"    Points: {len(series['points'])}")
            for j, point in enumerate(series['points']):
                # Scale point coordinates (assuming normalized coordinates)
                x, y = int(point[0] * img_width), int(point[1] * img_height)
                print(f"      Point {j+1}: ({x}, {y})")
    
    # Print detected elements
    print("\nDetected Elements:")
    print("-"*30)
    
    bboxes = instances.bboxes.cpu().numpy()
    scores = instances.scores.cpu().numpy()
    labels = instances.labels.cpu().numpy()
    
    # Group elements by class
    elements_by_class = {}
    for idx in range(len(bboxes)):
        score = scores[idx]
        if score < score_thr:
            continue
            
        label = labels[idx]
        class_name = class_names[label]
        x1, y1, x2, y2 = bboxes[idx].astype(int)
        
        if class_name not in elements_by_class:
            elements_by_class[class_name] = []
            
        elements_by_class[class_name].append({
            'confidence': score,
            'bbox': [x1, y1, x2, y2],
            'area': (x2-x1)*(y2-y1)
        })
    
    # Print grouped elements
    for class_name, elements in elements_by_class.items():
        print(f"\n{class_name.upper()}:")
        for i, elem in enumerate(elements):
            print(f"  Element {i+1}:")
            print(f"    Confidence: {elem['confidence']:.3f}")
            print(f"    Bounding Box: {elem['bbox']}")
            print(f"    Area: {elem['area']} pixels")
    
    print("\n" + "="*50)

def load_metadata(ann_file, img_filename):
    """Load metadata for a specific image from annotations file."""
    with open(ann_file, 'r') as f:
        annotations = json.load(f)
    
    for img_info in annotations['images']:
        if img_info['file_name'] == img_filename:
            return {
                'chart_type': img_info.get('chart_type', 'Unknown'),
                'plot_bb': img_info.get('plot_bb', None),
                'axes_info': img_info.get('axes_info', None),
                'data_series': img_info.get('data_series', None)
            }
    return None

def main():
    register_all_modules()

    # 1) Load your minimal model-only config
    cfg = Config.fromfile('cascade_rcnn_r50_fpn_test.py')

    # 2) Update model configuration to match training config
    cfg.model.update(dict(
        type='CustomCascadeWithMeta',
        chart_cls_head=dict(
            type='FCHead',
            in_channels=256,
            num_classes=5,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0)
        ),
        plot_reg_head=dict(
            type='RegHead',
            in_channels=256,
            out_dims=4,
            loss=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)
        ),
        axes_info_head=dict(
            type='RegHead',
            in_channels=256,
            out_dims=8,
            loss=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)
        ),
        data_series_head=dict(
            type='RegHead',
            in_channels=256,
            out_dims=2,
            max_points=50,
            loss=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)
        )
    ))

    # 3) Patch in a minimal test_dataloader so inference_detector can find a pipeline
    cfg.test_dataloader = {
        'dataset': {
            'type': 'ChartDataset',  # Use our custom dataset
            'data_root': os.getcwd(),
            'ann_file': 'test/annotations.json',
            'data_prefix': dict(img='test/'),
            'metainfo': {
                'classes': ['plot_area', 'title', 'x_axis', 'y_axis', 'legend']
            },
            'pipeline': [
                {'type': 'LoadImageFromFile'},
                {'type': 'LoadAnnotations', 'with_bbox': True, 'with_mask': False},
                {'type': 'Resize', 'scale': (1333, 800), 'keep_ratio': True},
                {'type': 'Normalize', 'mean': [123.675, 116.28, 103.53],
                                 'std': [58.395, 57.12, 57.375],
                                 'to_rgb': True},
                {'type': 'Pad', 'size_divisor': 32},
                {'type': 'PackChartInputs', 'meta_keys': (
                    'img_id', 'img_path', 'ori_shape', 'img_shape',
                    'pad_shape', 'scale_factor', 'chart_type',
                    'plot_bb', 'axes_info', 'data_series'
                )}
            ]
        }
    }

    # 4) Build model and load checkpoint
    model = MODELS.build(cfg.model)
    ckpt = torch.load('epoch_2.pth', map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # 5) Attach patched cfg so inference_detector will see it
    model.cfg = cfg

    # Get class names from metainfo
    class_names = list(cfg.test_dataloader['dataset']['metainfo']['classes'])

    # Process test images
    test_img_dir = 'test'  # or your actual test image directory
    test_img_dir_full = os.path.join(os.getcwd(), test_img_dir)
    img_files = [f for f in os.listdir(test_img_dir_full) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    out_dir = 'test_infer_results'
    os.makedirs(out_dir, exist_ok=True)

    for img_file in img_files:
        img_path = os.path.join(test_img_dir_full, img_file)
        print(f"\nProcessing {img_path}")
        
        # Load image and convert to tensor
        img = mmcv.imread(img_path)
        # Convert image to tensor and adjust dimensions
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()  # HWC to CHW
        # Create a proper data dict with both inputs and data_samples
        data_sample = DetDataSample()
        data_sample.set_metainfo({
            'img_shape': img.shape[:2],
            'ori_shape': img.shape[:2],
            'pad_shape': img.shape[:2],
            'scale_factor': 1.0,
            'flip': False,
            'flip_direction': None
        })
        data = {
            'inputs': [img_tensor],
            'data_samples': [data_sample]
        }
        img_tensor = model.data_preprocessor(data, False)['inputs']
        img_tensor = img_tensor.to(next(model.parameters()).device)
        img_metas = [{}]  # Dummy meta, can be improved if needed

        # Call model's simple_test directly to get predictions
        with torch.no_grad():
            results = model.simple_test(img_tensor, img_metas)
        pred_result = results[0]  # Assume batch size 1

        # Print all predicted metadata
        print("\nPredicted Metadata from model.simple_test:")
        for k, v in pred_result.items():
            print(f"{k}: {v}")

        # Extract predicted metadata for visualization
        pred_chart_type = pred_result.get('chart_type', 'Unknown')
        pred_plot_bb = pred_result.get('plot_bb', None)
        pred_axes_info = pred_result.get('axes_info', None)
        pred_data_series = pred_result.get('data_series', None)

        # Compose a meta_info dict for visualization
        meta_info = {
            'chart_type': pred_chart_type,
            'plot_bb': pred_plot_bb,
            'axes_info': pred_axes_info,
            'data_series': [
                {'type': 'line', 'points': [[pt['x'], pt['y']] for pt in pred_data_series]}
            ] if pred_data_series is not None and len(pred_data_series) > 0 else []
        }

        # For detection results, use pred_result['instances'] if available, else fallback
        # Here, we use the original detection pipeline for bounding boxes
        data_sample = inference_detector(model, img_path)

        out_path = os.path.join(out_dir, f"labeled_{os.path.splitext(os.path.basename(img_file))[0]}.png")
        visualize_detections(img, data_sample.pred_instances, class_names, meta_info, score_thr=0.3, out_path=out_path, pred_data_series=pred_data_series)
        print_detection_results(data_sample.pred_instances, class_names, meta_info, score_thr=0.3)

if __name__ == '__main__':
    main()
 