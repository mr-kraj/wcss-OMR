import argparse
import sys
import os
import cv2
import supervision as sv
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", "-s", required=True, help="Image path")
    parser.add_arguement("--preprocess", "-p", required=False, default=True, help="Apply preprocessing (remove barlines)")
    model = YOLO(str(model_path))

    print("Scanning image: ", args.source)

    results = model.predict(
        source=str(source_path),
        imgsz=args.imgsz,
        save=False, 
        device=args.device,
        verbose=True,
    )

    boxes = results[0].boxes
    image = results[0].orig_img.copy()

    labels = [f"{model.model.names[int(cls)]} {conf:.2f}"for cls, conf in zip(boxes.cls, boxes.conf)]
    detections = sv.Detections.from_ultralytics(results[0])

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_scale=float(args.label_size), text_thickness=1)
    
    image_boxes = box_annotator.annotate(scene=image,detections=detections)
    blending_mask = image_boxes.copy()
    image_labels = label_annotator.annotate(scene=image_boxes,detections=detections)

    alpha = 0.5
    final_image = cv2.addWeighted(image_labels, alpha, blending_mask, 1 - alpha, 0)

    cv2.imwrite("runs/predict/results.png", final_image)
if __name__ == "__main__":
    main()
