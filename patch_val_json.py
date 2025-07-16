import json
import os
from datetime import datetime

data_root = '/content/drive/MyDrive/Research Summer 2025/Dense Captioning Toolkit/CHART-DeMatch/legend_data/'
orig = os.path.join(data_root, 'annotations_JSON/val.json')
new_file = os.path.join(data_root, 'annotations_JSON/val_with_info.json')

# Load original JSON
with open(orig, 'r') as f:
    ann = json.load(f)

# Inject a complete "info" block
ann['info'] = {
    "description": "Chart element detection dataset",
    "version": "1.0",
    "year": 2025,
    "contributor": "Chart-DeMatch",
    "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

# Ensure all required top-level keys exist
for key in ('licenses', 'images', 'annotations', 'categories'):
    ann.setdefault(key, [])

# Ensure categories include all our classes
if not ann['categories']:
    # Chart Elements
    chart_elements = [
        {"id": 1, "name": "plot_area", "supercategory": "chart_element"},
        {"id": 2, "name": "line_segment", "supercategory": "chart_element"},
        {"id": 3, "name": "axis_tick", "supercategory": "chart_element"},
        {"id": 4, "name": "axis_label", "supercategory": "chart_element"},
        {"id": 5, "name": "chart_title", "supercategory": "chart_element"},
        {"id": 6, "name": "scatter_point", "supercategory": "chart_element"},
        {"id": 7, "name": "bar_segment", "supercategory": "chart_element"},
        {"id": 8, "name": "dot_point", "supercategory": "chart_element"}
    ]
    
    # Text Roles
    text_roles = [
        {"id": 9, "name": "tick_label", "supercategory": "text_role"},
        {"id": 10, "name": "axis_title", "supercategory": "text_role"},
        {"id": 11, "name": "chart_title", "supercategory": "text_role"},
        {"id": 12, "name": "legend_title", "supercategory": "text_role"},
        {"id": 13, "name": "legend_label", "supercategory": "text_role"}
    ]
    
    # Chart Types
    chart_types = [
        {"id": 14, "name": "line", "supercategory": "chart_type"},
        {"id": 15, "name": "scatter", "supercategory": "chart_type"},
        {"id": 16, "name": "dot", "supercategory": "chart_type"},
        {"id": 17, "name": "vertical_bar", "supercategory": "chart_type"},
        {"id": 18, "name": "horizontal_bar", "supercategory": "chart_type"}
    ]
    
    ann['categories'] = chart_elements + text_roles + chart_types

# Add metadata to each image while preserving existing data
for img in ann['images']:
    # Preserve existing fields
    existing_data = {k: v for k, v in img.items()}
    
    # Add required metadata fields if they don't exist
    if 'chart_type' not in img:
        img['chart_type'] = None
    if 'plot_bb' not in img:
        img['plot_bb'] = None
    if 'axes_info' not in img:
        img['axes_info'] = None
    
    # Ensure required COCO fields exist
    if 'license' not in img:
        img['license'] = 0
    if 'flickr_url' not in img:
        img['flickr_url'] = ""
    if 'coco_url' not in img:
        img['coco_url'] = ""
    if 'date_captured' not in img:
        img['date_captured'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Write to new file
with open(new_file, 'w') as f:
    json.dump(ann, f, indent=2)

print("✅ Created val_with_info.json with all required metadata fields while preserving existing data. Ready to train.") 