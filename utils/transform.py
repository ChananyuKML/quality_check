import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import sys
import argparse # Import argparse
import cv2
import numpy as np
from typing import List, Tuple

class CustomFeatureTransform(object):
    def __init__(self, canny_threshold1=50, canny_threshold2=150,
                 adaptive_block_size=11, 
                 adaptive_C=2):
        self.canny_threshold1 = canny_threshold1
        self.canny_threshold2 = canny_threshold2

        if adaptive_block_size % 2 == 0:
            adaptive_block_size += 1 
        self.adaptive_block_size = adaptive_block_size
        self.adaptive_C = adaptive_C

    def __call__(self, object):
        np_img = np.array(object)
        gray_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        
        ch0_gray = gray_img.copy()
        ch0_binary = cv2.adaptiveThreshold(
            gray_img,
            255,  # Max value to assign
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # Method: use a weighted sum (Gaussian)
            cv2.THRESH_BINARY,              # Threshold type
            self.adaptive_block_size,       # Neighborhood size
            self.adaptive_C                 # Constant C
        )

        f = np.fft.fft2(gray_img)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.log(np.abs(fshift) + 1)
        ch1_fft = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        ch2_edges = cv2.Canny(gray_img, self.canny_threshold1, self.canny_threshold2)

        combined_features = np.stack([ch2_edges, ch2_edges, ch2_edges], axis=2)
        tensor = torch.from_numpy(combined_features.astype(np.float32) / 255.0)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        
        return tensor

def crop_objects(image: np.ndarray, boxes: List[Tuple[int, int, int, int, int]]) -> List[np.ndarray]:
    """Crops regions defined by bounding boxes from the original image."""
    cropped_images = []
    
    for class_id, x_min, y_min, x_max, y_max in boxes:
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
        cropped_img = image[y_min:y_max, x_min:x_max]
        if cropped_img.size > 0:
            cropped_images.append(cropped_img)
    
    return cropped_images