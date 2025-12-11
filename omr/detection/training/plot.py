import pandas as pd
import matplotlib.pyplot as plt

# Load your CSV data
csv_path = "lustre/pd01/hpc-katjar5048-1762193819/OMR/omr/detection/training/training_runs/4340805/Actual YOLO Training4340805/results.csv"  # replace with your CSV file path
df = pd.read_csv(csv_path)

# Epochs
epochs = df['epoch']

# Plot training and validation losses
plt.figure(figsize=(16, 6))
plt.plot(epochs, df['train/box_loss'], label='train/box_loss')
plt.plot(epochs, df['train/cls_loss'], label='train/cls_loss')
plt.plot(epochs, df['train/dfl_loss'], label='train/dfl_loss')
plt.plot(epochs, df['val/box_loss'], label='val/box_loss', linestyle='--')
plt.plot(epochs, df['val/cls_loss'], label='val/cls_loss', linestyle='--')
plt.plot(epochs, df['val/dfl_loss'], label='val/dfl_loss', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('YOLO Training & Validation Losses')
plt.legend()
plt.grid(True)
plt.show()

# Plot metrics
plt.figure(figsize=(16, 6))
plt.plot(epochs, df['metrics/precision(B)'], label='Precision')
plt.plot(epochs, df['metrics/recall(B)'], label='Recall')
plt.plot(epochs, df['metrics/mAP50(B)'], label='mAP50')
plt.plot(epochs, df['metrics/mAP50-95(B)'], label='mAP50-95')
plt.xlabel('Epoch')
plt.ylabel('Metric Value')
plt.title('YOLO Metrics')
plt.legend()
plt.grid(True)
plt.show()