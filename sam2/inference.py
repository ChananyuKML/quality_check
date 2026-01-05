from ultralytics import SAM
import argparse

parser = argparse.ArgumentParser(description="Configuration for training process")
parser.add_argument("img", type=str)
parser.add_argument("--pt", type=str, default="sam2.1_t")
args = parser.parse_args()

# Load pre-trained model
model = SAM(f"{args.pt}.pt")

# Inference Model
results = model(f"{args.img}")

# Iterate through results (one result per image in the batch)
for result in results:
    masks = result.masks  
    boxes = result.boxes  
    mask_data = masks.data 
    box_data = boxes.xyxy