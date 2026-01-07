import cv2
import numpy as np
import src.similarity_resnet as simr
import sam2.module as sam
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

feature_extractor = simr.FeatureExtractor()

class runOptions(BaseModel):
    img1: str = "img/pc.jpg"
    img2: str = "img/test-1.png"
    img: str = "img/pc.jpg"
    tgt_width: int = 640
    tgt_height: int = 640

def compare_similarity(img1, img2):
    f_img1 = simr.get_image_features(img1, feature_extractor)
    f_img2 = simr.get_image_features(img2, feature_extractor)
    sim = simr.calculate_similarity(f_img1, f_img2)
    return f"(Image 1 vs Image 2): {sim:.4f}"

@app.post("/sim")
def run(data: runOptions):
    result = compare_similarity(data.img1, data.img2)     
    return {"result": result}

@app.post("/sam")
def run(data: runOptions):
    result = sam.run(img=data.img)
    return {"result": result}