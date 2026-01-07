from ultralytics import SAM
from PIL import Image
import argparse

def run(pt="sam2.1_t", img="img\pc.jpg", prompt="car"):
    # with Image.open(img) as image:
    #     # The .size attribute returns a tuple (width, height)
    #     width, height = image.size
    model = SAM(f"{pt}.pt")
    results = model(f"{img}", save=True)
    for result in results:
        masks = result.masks  
        boxes = result.boxes  
        mask_data = masks.data 
        box_data = boxes.xyxy
        
    json_results = results[0].to_json()
    return json_results

if __name__=="__main__":
    run()