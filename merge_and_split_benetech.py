import os
import json
import random
import re
from PIL import Image
import time

def fast_jpeg_size(path):
    """Quick width/height from a JPEG's SOF0 marker."""
    with open(path, "rb") as f:
        data = f.read(2048)
    pos = data.find(b"\xFF\xC0")
    if pos < 0:
        raise ValueError(f"SOF0 marker not found in {path}")
    h = int.from_bytes(data[pos+5:pos+7], "big")
    w = int.from_bytes(data[pos+7:pos+9], "big")
    return w, h

def get_image_size(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        return fast_jpeg_size(path)
    else:
        with Image.open(path) as img:
            return img.width, img.height

def merge_and_split_benetech(seed: int = 42):
    # ─── Point this at your Drive folder ──────────────────────────────────
    BASE_DIR = (
        "/content/drive/MyDrive/Research Summer 2025/"
        "Dense Captioning Toolkit/CHART-DeMatch/legend_data"
    )
    ANN_DIR       = os.path.join(BASE_DIR, "train", "annotations")
    IMG_DIR       = os.path.join(BASE_DIR, "train", "images")
    OUTPUT_FOLDER = os.path.join(BASE_DIR, "annotations_JSON")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Verify directories exist and are accessible
    if not os.path.exists(ANN_DIR):
        raise OSError(f"Annotations directory does not exist: {ANN_DIR}")
    if not os.path.exists(IMG_DIR):
        raise OSError(f"Images directory does not exist: {IMG_DIR}")

    # 1) build a map json_basename → actual image filename
    print("Building image filename map...")
    image_map = {}
    for fn in os.listdir(IMG_DIR):
        name, ext = os.path.splitext(fn)
        if ext.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        # grab the leading hex chunk
        m = re.match(r"^([0-9a-f]+)", name)
        if not m:
            continue
        hexid = m.group(1)
        # in case of duplicates, keep the first
        image_map.setdefault(hexid, fn)
    print(f"Found {len(image_map)} valid image files")

    # 2) gather only JSON files that have corresponding valid images
    print("Gathering JSON files for valid images...")
    all_jsons = []
    for base in image_map.keys():
        json_file = f"{base}.json"
        json_path = os.path.join(ANN_DIR, json_file)
        if os.path.exists(json_path):
            all_jsons.append(json_file)
    
    print(f"Found {len(all_jsons)} valid JSON files (matching valid images)")
    random.seed(seed)
    random.shuffle(all_jsons)
    split_idx = int(0.7 * len(all_jsons))
    splits = {
        "train": all_jsons[:split_idx],
        "val":   all_jsons[split_idx:]
    }

    # 3) categories
    categories = [
        {"id": 1, "name": "plot_area"},
        {"id": 2, "name": "line_segment"},
        {"id": 3, "name": "axis_tick"},
        {"id": 4, "name": "axis_label"},
        {"id": 5, "name": "chart_title"},
    ]
    cat2id = {c["name"]: c["id"] for c in categories}

    for split_name, json_files in splits.items():
        print(f"\nProcessing {split_name} split...")
        coco = {"images": [], "annotations": [], "categories": categories}
        img_id_map = {}
        next_img_id = 1
        next_ann_id = 1
        processed_count = 0
        total_files = len(json_files)

        for jf in json_files:
            try:
                base = os.path.splitext(jf)[0]
                img_fn = image_map.get(base)
                if img_fn is None:
                    continue

                img_path = os.path.join(IMG_DIR, img_fn)
                w, h = get_image_size(img_path)

                # Load original annotation
                ann_path = os.path.join(ANN_DIR, jf)
                try:
                    with open(ann_path, 'r') as f:
                        ann = json.load(f)
                except (OSError, json.JSONDecodeError) as e:
                    print(f"Error reading {jf}: {e}")
                    continue

                # register image with additional metadata
                if img_fn not in img_id_map:
                    img_id_map[img_fn] = next_img_id
                    image_info = {
                        "id": next_img_id,
                        "file_name": img_fn,
                        "width": w,
                        "height": h,
                        "source": ann.get("source", ""),
                        "chart_type": ann.get("chart-type", ""),
                        "plot_bb": ann.get("plot-bb", {}),
                        "data_series": ann.get("data-series", []),
                        "axes_info": {
                            "x-axis": {
                                "tick-type": ann.get("axes", {}).get("x-axis", {}).get("tick-type", ""),
                                "values-type": ann.get("axes", {}).get("x-axis", {}).get("values-type", "")
                            },
                            "y-axis": {
                                "tick-type": ann.get("axes", {}).get("y-axis", {}).get("tick-type", ""),
                                "values-type": ann.get("axes", {}).get("y-axis", {}).get("values-type", "")
                            }
                        }
                    }
                    coco["images"].append(image_info)
                    next_img_id += 1
                img_id = img_id_map[img_fn]

                def add_bbox(x0, y0, ww, hh, cat_name, additional_info=None):
                    nonlocal next_ann_id
                    if ww > 0 and hh > 0:
                        annotation = {
                            "id": next_ann_id,
                            "image_id": img_id,
                            "category_id": cat2id[cat_name],
                            "bbox": [x0, y0, ww, hh],
                            "area": ww * hh,
                            "iscrowd": 0,
                            "segmentation": []
                        }
                        if additional_info:
                            annotation.update(additional_info)
                        coco["annotations"].append(annotation)
                        next_ann_id += 1

                # plot_area
                bb = ann.get("plot-bb", {})
                if all(k in bb for k in ("x0","y0","width","height")):
                    add_bbox(bb["x0"], bb["y0"], bb["width"], bb["height"], "plot_area")

                # line_segment
                for line in ann.get("visual-elements", {}).get("lines", []):
                    xs = [p["x"] for p in line if "x" in p]
                    ys = [p["y"] for p in line if "y" in p]
                    if xs and ys:
                        x0, y0 = min(xs), min(ys)
                        x1, y1 = max(xs), max(ys)
                        add_bbox(x0, y0, x1-x0, y1-y0, "line_segment", {
                            "line_points": line,
                            "visual_element_type": "line"
                        })

                # text roles
                for t in ann.get("text", []):
                    poly = t.get("polygon", {})
                    if all(k in poly for k in ("x0","x1","x2","x3","y0","y1","y2","y3")):
                        xs = [poly[f"x{i}"] for i in range(4)]
                        ys = [poly[f"y{i}"] for i in range(4)]
                        x0, y0 = min(xs), min(ys)
                        ww, hh = max(xs)-x0, max(ys)-y0
                        role = t.get("role","")
                        additional_info = {
                            "text": t.get("text", ""),
                            "role": role,
                            "polygon": poly,
                            "text_id": t.get("id", -1)
                        }
                        if role in ("tick_label","axis_title"):
                            add_bbox(x0, y0, ww, hh, "axis_label", additional_info)
                        elif role == "chart_title":
                            add_bbox(x0, y0, ww, hh, "chart_title", additional_info)

                # axis ticks
                for axis in ("x-axis","y-axis"):
                    for tick in ann.get("axes", {}).get(axis, {}).get("ticks", []):
                        pt = tick.get("tick_pt", {})
                        if "x" in pt and "y" in pt:
                            add_bbox(pt["x"]-2.5, pt["y"]-2.5, 5, 5, "axis_tick", {
                                "axis": axis,
                                "tick_id": tick.get("id", -1),
                                "tick_pt": pt,
                                "tick_type": ann.get("axes", {}).get(axis, {}).get("tick-type", ""),
                                "values_type": ann.get("axes", {}).get(axis, {}).get("values-type", "")
                            })

                processed_count += 1
                if processed_count % 10 == 0:  # Print progress every 10 files
                    print(f"Processed {processed_count}/{total_files} files ({(processed_count/total_files)*100:.1f}%)")

            except Exception as e:
                print(f"Error processing {jf}: {e}")
                continue

        out_path = os.path.join(OUTPUT_FOLDER, f"{split_name}.json")
        with open(out_path, "w") as f:
            json.dump(coco, f, indent=2)

        # sanity check
        imgs = json.load(open(out_path))["images"]
        assert all("width" in im and "height" in im for im in imgs), "missing size!"
        print(f"✅ Wrote {out_path}: {len(imgs)} images, {len(coco['annotations'])} annotations")

if __name__ == "__main__":
    merge_and_split_benetech() 