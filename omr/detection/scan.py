import argparse
import sys
import os
import cv2
import supervision as sv
import json
import torch
from ultralytics import YOLO

import omr.preprocessing.segmenter as segmenter

BASE_PATH = os.environ.get("BASE_PATH", ".")
CONFIG = parse_CONFIG(BASE_PATH + "/omr/detection/scanner_CONFIG.json")

def extract_labels_for_boxes(results, model):
    boxes = results.boxes
    labels_with_confidence = []
    raw_labels = []
    
    for cls, conf in zip(boxes.cls, boxes.conf):
        label = model.model.names[int(cls)]

        labels_with_confidence.append([f"{label} {conf:.2f}"])
        raw_labels.append(label)
    return labels_with_confidence, raw_labels

def parse_to_list(results, labels):
    boxes = results.boxes
    parsed = [{"class": labels[i], "bounding_box": boxes.xyxy[i].tolist()} for i in range(len(labels))]

    return parsed

def parse_config(path):
    # TODO FIX THIS!!!!
    config = json.load(path)

    for key in CONFIG:
        if isinstance(config[key], str):
            config[key] = BASE_PATH + config[key]

    return config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", "-i", required=True, help="Image path")
    args = parser.parse_args()

    model = YOLO(CONFIG["modelPath"])

    processed = segmenter.segment_music_sheet(args.image_path, CONFIG["segmenter_threshold"], CONFIG["segmenter_tolerance"]).staff_regions_no_lines
    results = model.predict(
        source=processed,
        save=False, 
        device=0 if torch.cuda.is_available() else 'cpu',
        verbose=False
    )

    labels = extract_labels_for_boxes(results[0], model)

    # For debugging purposes only - save annotated images
    
    import omr.detection.scanner.scan as scan
    for i in range(len(results)):
        os.makedirs(BASE_PATH + "/omr/detection/.temp/segmented/", exist_ok=True)
        scan.save_file(results[i], labels, BASE_PATH + "/omr/detection/.temp/segmented/result_" + str(i) + ".png")
    
    return [parse_to_list(result, labels) for result in results]

if __name__ == "__main__":
    main()