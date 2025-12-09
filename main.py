import src.similarity_resnet as simr
from fastapi import FastAPI
from pydantic import BaseModel

feature_extractor = simr.FeatureExtractor
get_feature = simr.get_image_features(feature_extractor())
cal_sim = simr.calculate_similarity()

def compare_similarity(img1, img2):
    f_img1 = get_feature(img1)
    f_img2 = get_feature(img2)
    sim = cal_sim(f_img1, f_img2)
    return f"(Image 1 vs Image 2): {sim:.4f}"

class runOptions(BaseModel):
    img1: str = "img/pc.jpg"
    img2: str = "img/test-1.png"

@app.post("/run")
def run(data: runOptions):
    result = compare_similarity(data.img1, data.img2)     
    return {"result": result}