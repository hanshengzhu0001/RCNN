import json
import os
import numpy as np
from mmdet.datasets import CocoDataset
from mmengine.config import Config
from google.colab import drive

def inspect_dataset(ann_file, data_prefix, data_root):
    print(f"\nInspecting dataset: {ann_file}")
    if not os.path.exists(ann_file):
        print(f"Error: Annotation file {ann_file} does not exist!")
        return
    
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    print(f"Number of images: {len(data['images'])}")
    print(f"Number of annotations: {len(data['annotations'])}")
    
    # Check if images exist
    missing_images = []
    for img in data['images']:
        img_path = os.path.join(data_root, data_prefix['img'], img['file_name'])
        if not os.path.exists(img_path):
            missing_images.append(img['file_name'])
    
    if missing_images:
        print(f"Warning: {len(missing_images)} images are missing!")
        print("First few missing images:", missing_images[:5])
    else:
        print("All images exist in the specified directory")

    # Analyze annotations
    print("\nAnalyzing annotations:")
    # Count annotations per image
    anns_per_image = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in anns_per_image:
            anns_per_image[img_id] = 0
        anns_per_image[img_id] += 1
    
    print(f"Number of images with annotations: {len(anns_per_image)}")
    print(f"Number of images without annotations: {len(data['images']) - len(anns_per_image)}")
    
    # Check annotation sizes and masks
    bbox_sizes = []
    has_mask = 0
    no_mask = 0
    for ann in data['annotations']:
        if 'bbox' in ann:
            bbox = ann['bbox']
            if len(bbox) == 4:  # [x, y, width, height]
                bbox_sizes.append(bbox[2] * bbox[3])  # width * height
        
        if 'segmentation' in ann:
            if ann['segmentation']:
                has_mask += 1
            else:
                no_mask += 1
    
    print("\nMask statistics:")
    print(f"Annotations with masks: {has_mask}")
    print(f"Annotations without masks: {no_mask}")
    
    if bbox_sizes:
        print("\nBounding box size statistics:")
        print(f"Min size: {min(bbox_sizes):.2f}")
        print(f"Max size: {max(bbox_sizes):.2f}")
        print(f"Mean size: {np.mean(bbox_sizes):.2f}")
        print(f"Number of boxes smaller than 1 pixel: {sum(1 for s in bbox_sizes if s < 1)}")
    
    # Try creating dataset with different filter settings
    print("\nTrying different filter settings:")
    
    # Try without filtering
    try:
        dataset = CocoDataset(
            ann_file=ann_file,
            data_prefix=data_prefix,
            data_root=data_root,
            filter_cfg=dict(
                filter_empty_gt=False,
                min_size=0,
                bbox_min_size=0,
                mask_min_size=0
            )
        )
        print(f"Without filtering: {len(dataset)} samples")
    except Exception as e:
        print(f"Error without filtering: {str(e)}")
    
    # Try with current settings but without mask filtering
    try:
        dataset = CocoDataset(
            ann_file=ann_file,
            data_prefix=data_prefix,
            data_root=data_root,
            filter_cfg=dict(
                filter_empty_gt=True,
                min_size=1,
                bbox_min_size=1,
                mask_min_size=0  # Don't filter by mask size
            )
        )
        print(f"With bbox filtering only: {len(dataset)} samples")
    except Exception as e:
        print(f"Error with bbox filtering: {str(e)}")
    
    # Try with current settings
    try:
        dataset = CocoDataset(
            ann_file=ann_file,
            data_prefix=data_prefix,
            data_root=data_root,
            filter_cfg=dict(
                filter_empty_gt=True,
                min_size=1,
                bbox_min_size=1,
                mask_min_size=1
            )
        )
        print(f"With current filtering: {len(dataset)} samples")
    except Exception as e:
        print(f"Error with current filtering: {str(e)}")

if __name__ == '__main__':
    # Mount Google Drive if not already mounted
    if not os.path.exists('/content/drive'):
        print("Mounting Google Drive...")
        drive.mount('/content/drive')
    
    # Update these paths according to your Colab environment
    data_root = '/content/drive/MyDrive/Research Summer 2025/Dense Captioning Toolkit/CHART-DeMatch/legend_data/'
    
    print("\n=== Dataset Inspection ===")
    print(f"Checking data root: {data_root}")
    if not os.path.exists(data_root):
        print(f"Error: Data root directory does not exist!")
    else:
        print("Data root directory exists")
        print("Contents of data root:")
        print(os.listdir(data_root))
        
        inspect_dataset(
            os.path.join(data_root, 'annotations_JSON/train.json'),
            {'img': 'train/images/'},
            data_root
        )
        inspect_dataset(
            os.path.join(data_root, 'annotations_JSON/val.json'),
            {'img': 'train/images/'},
            data_root
        )
    print("========================\n") 