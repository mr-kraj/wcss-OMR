import argparse
import json
from ultralytics import YOLO

def load_and_convert_to_png(image_path):
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_arg("--input")

    args = parser.parse_args()
    image = load_and_convert_to_png(args.input)

    with open("modelConfig.json", 'r') as f:
        config = json.load(f)

    model = YOLO(config["modelPath"])
    results = model(image)

    print(results)

if __name__ == "__main__":
    main()