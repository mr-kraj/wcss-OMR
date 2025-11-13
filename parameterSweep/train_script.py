import csv
import argparse
import os
from ultralytics import YOLO

def load_config(csv_path, idx):
    with open(csv_path, newline='') as f:
        reader = list(csv.DictReader(f))
    row = reader[int(idx)]
    for k, v in row.items():
        try:
            row[k] = float(v) if "." in v else int(v)
        except ValueError:
            pass
    return row

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv")
    parser.add_argument("--idx")
    parser.add_argument("--data")
    parser.add_argument("--model")
    parser.add_argument("--workdir")
    parser.add_argument("--results_csv")
    args = parser.parse_args()

    cfg = load_config(args.csv, args.idx)

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=int(cfg.get("epochs", 100)),
        batch=int(cfg.get("batch", 16)),
        lr0=float(cfg.get("lr", 1e-3)),
        imgsz=int(cfg.get("imgsz", 640)),
        project=args.workdir,
        name=f"run_{args.idx}",
        device=0,
    )

    columns = ["epochs", "batch", "lr", "imgsz"] + list(results.metrics.keys())
    config_dict = {"epochs": int(cfg.get("epochs", 100)),
                       "batch": int(cfg.get("batch", 16)),
                       "lr": float(cfg.get("lr", 1e-3)),
                       "imgsz": int(cfg.get("imgsz", 640))}
    
    results_dict = {**config_dict, **results.metrics}
    file_exists = os.path.exists(args.results_csv)

    with open(args.results_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)

        if not file_exists:
            writer.writeheader()

        writer.writerow(results_dict)

if __name__ == "__main__":
    main()