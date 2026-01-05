from ultralytics import SAM
from PIL import Image
import argparse

def run(pt="sam2.1_t", img="img\pc.jpg", prompt="car"):
    with Image.open(img) as image:
        # The .size attribute returns a tuple (width, height)
        width, height = image.size
    model = SAM(f"{pt}.pt")
    results = model(f"{img}",points=[width/2, height/2], labels=[1], save=True)
    print(results)
    for result in results:
        boxes_data = result.boxes.xywh # or .xyxy, .numpy(), etc. 
        print(f"box data: {boxes_data}")
        mask_data = result.masks # or .xyxy, .numpy(), etc. 
        print(f"box data: {boxes_data}")
        print(f"mask data: {mask_data}")
    return boxes_data

if __name__=="__main__":
    run()