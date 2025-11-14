import argparse
import numpy as np
import cv2
import torchvision.models as models
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from utils.transform import CustomFeatureTransform
import os

transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x

def reproject(src,ref):
    ref_width, ref_height = ref_img.size
    src_width, src_height = src_img.size
    
    src_arr = np.array(src)
    ref_arr = np.array(ref)

    pts_src = np.array([
        [0, 0],         # Top-left
        [src_width, 0],  # Top-right
        [src_width, src_height], # Bottom-right
        [0, src_height]  # Bottom-left
    ], dtype="float32")
   
    pts_dst = np.array([
        [0, 0],         # Top-left
        [ref_width, 0],  # Top-right
        [ref_width, ref_height], # Bottom-right
        [0, ref_height]  # Bottom-left
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
    rep_src = cv2.warpPerspective(src_arr, matrix, (ref_width, ref_height))
    return Image.fromarray(rep_src)


def get_image_features(image_path, model):
    image_tensor = preprocess_image(image_path)
    
    if image_tensor is None:
        return None
    
    model.eval()
    with torch.no_grad(): # Disable gradient calculation for inference
        features = model(image_tensor)
    return features

def calculate_similarity(features1, features2):
    similarity_score = F.cosine_similarity(features1, features2, dim=1)
    return similarity_score.item()

parser = argparse.ArgumentParser(description="A simple script demonstrating argparse.")

parser.add_argument('--boxes', type=str, default="dataset/components/labels/1377c66d-front_ref.txt", help='Path to label file')
parser.add_argument('--ref', type=str, default="dataset/components/images/1377c66d-front_ref.jpg", help='Path to label file')
parser.add_argument('img', type=str, default="img/front_view_output.jpg", help='Path to label file')
args = parser.parse_args()

ref_img = Image.open(args.ref).convert('RGB')
src_img = Image.open(args.img).convert('RGB')
src_img = reproject(src_img,ref_img)
img_width, img_height = ref_img.size
model = FeatureExtractor()

src_list = []
ref_list = []
bounding_boxes = []

file_path = args.boxes
with open(file_path, 'r') as f:
    for line in f:
        # Split the line by spaces or commas and convert to float
        parts = line.strip().replace(',', ' ').split()
        label, x_center, y_center, width, height = map(float, parts)
        #top-left > top-right > bottom-right > bottom-left
        bounding_boxes.append([int(label),
                                int(img_width*(x_center-(width/2))), 
                                int(img_height*(y_center-(height/2))), 
                                int(img_width*(x_center+(width/2))), 
                                int(img_height*(y_center+(height/2)))])


for i, box in enumerate(bounding_boxes):
    src_patch = src_img.crop((box[1], box[2], box[3], box[4]))
    ref_patch = ref_img.crop((box[1], box[2], box[3], box[4]))

    src_list.append(src_patch)
    ref_list.append(ref_patch)

    with torch.no_grad():
        src_feature = model(transform(src_patch).unsqueeze(0))
        ref_feature = model(transform(ref_patch).unsqueeze(0))
    sim_score = F.cosine_similarity(src_feature, ref_feature, dim=1).item()
    print(f"At Box : {i}, similarity_score is equal to {sim_score:.2f}")

    filename = f"{i+1}"
    ref_patch.save(filename, format="JPEG")
    print(f"Saved: {filename}")
