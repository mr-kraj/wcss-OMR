import argparse
import sys
import os
import cv2
import supervision as sv
import json
import torch
from ultralytics import YOLO

#this refuses to work nicely on WCSS... to be changed
sys.path.insert(0, "/lustre/pd03/hpc-katjar5048-1762193819/OMR/")
print(sys.path)
import omr.preprocessing.segmenter as segmenter
#import omr.preprocessing.segmenter as segmenter

def extract_labels_for_boxes(results, model):
    boxes = results.boxes
    labels_with_confidence = []
    raw_labels = []
    
    for cls, conf in zip(boxes.cls, boxes.conf):
        label = model.model.names[int(cls)]

        labels_with_confidence.append([f"{label} {conf:.2f}"])
        raw_labels.append(label)
    return labels_with_confidence, raw_labels

def save_file(result, labels, file_path):
    image = result.orig_img.copy()

    detections = sv.Detections.from_ultralytics(result)

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
    
    image_boxes = box_annotator.annotate(scene=image,detections=detections)
    blending_mask = image_boxes.copy()
    image_labels = label_annotator.annotate(scene=image_boxes,detections=detections)

    alpha = 0.5
    final_image = cv2.addWeighted(image_labels, alpha, blending_mask, 1 - alpha, 0)

    cv2.imwrite(file_path, final_image)

def parse_to_list(results, labels):
    boxes = results.boxes
    parsed = [{"class": labels[i], "bounding_box": boxes.xyxy[i].tolist()} for i in range(len(labels))]

    return parsed

def preprocess(image_path):
    preprocessed = segmenter.preprocess_image(image_path)
    no_lines = segmenter.remove_staff_lines(preprocessed, segmenter.detect_staff_lines(preprocessed))

    file_name = image_path.split("/")[-1]
    cv2.imwrite("scanner/preprocessing/" + file_name, no_lines)
    return "scanner/preprocessing/" + file_name

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", "-i", required=True, help="Image path")
    parser.add_argument("--preprocess", "-p", action=argparse.BooleanOptionalAction, required=False, default=True, help="Apply preprocessing? (remove barlines)")
    parser.add_argument("--file-save", "-f", action=argparse.BooleanOptionalAction, required=False, default=False, help="Save result to file?")
    parser.add_argument("--file-path", "-fp", required=False, default="scanner/results/", help="File path to save result")
    args = parser.parse_args()

    config = json.load(open("scanner/modelConfig.json"))
    model = YOLO(config["modelPath"])

    if args.preprocess:
        print("Preprocessing...")
        new_image_path = preprocess(args.image_path)
        args.image_path = new_image_path
    
    print("Scanning image: ", args.image_path)

    results = model.predict(
        source=str(args.image_path),
        save=False, 
        device=0 if torch.cuda.is_available() else 'cpu',
        verbose=False
    )

    labels = extract_labels_for_boxes(results[0], model)

    if args.file_save:
        if(args.file_path == "scanner/results/"):
            os.makedirs("scanner/results/", exist_ok=True)
            file_name = args.image_path.split("/")[-1]
            args.file_path = args.file_path + file_name[:file_name.index(".")] + "_result" + file_name[file_name.index("."):]

        save_file(results[0], labels[0], args.file_path)
        print("Result saved to: ", args.file_path)

    return parse_to_list(results[0], labels[1])

if __name__ == "__main__":
    main()
