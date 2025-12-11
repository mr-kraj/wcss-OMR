import json
import argparse
from pathlib import Path
from ultralytics import YOLO

CONFIG_PATH = "/lustre/pd01/hpc-katjar5048-1762193819/OMR/omr/detection/training/training_config.json"
CONFIG = json.load(open(CONFIG_PATH, "r"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", "-i", required=True, help="Training job id")
    parser.add_argument("--gpu-count", "-g", required=True, help="Number of GPUs")
    args = parser.parse_args()

    model_path = CONFIG["model_init_weights_path"] if CONFIG["model_init_weights_path"] != "" else CONFIG["base_model_path"]

    model = YOLO(model_path)
    print("Training YOLO with: ", model_path)
    results = model.train(
        data=CONFIG["yaml_path"],
        epochs=CONFIG["epochs"],
        batch=CONFIG["batch"],
        lr0=CONFIG["lr"],
        imgsz=CONFIG["imgsz"],
        project=Path(CONFIG["training_dir_global_path"], CONFIG["training_result_dir_path"], args.id),
        name=f"Actual YOLO Training" + str(args.id),
        workers=CONFIG["workers"],
        device=list(range(int(args.gpu_count))) if args.gpu_count != -1 else "cpu",
        optimizer=CONFIG["optimizer"],
        save=True,
        cos_lr=False,
        conf=0.8)

if __name__ == "__main__":
    main()