# Step 1: Import necessary libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Compare the visual similarity of three images using ResNet50.")

parser.add_argument("image", default="test_output/pc.jpg",help="Path to the first image file.")
args = parser.parse_args()
# Step 3: Read the uploaded image
image_path = f"{args.image}"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Step 4: Apply Gaussian Blur to reduce noise
blurred_image = cv2.GaussianBlur(image, (5, 5), 1)

# Step 5: Apply Canny edge detector
edges = cv2.Canny(blurred_image, 0, 250)

# Step 6: Display the original image and the edge-detected image
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.title('Original Image')
# plt.imshow(image, cmap='gray')
# plt.axis('off')
# 
# plt.subplot(1, 2, 2)
plt.title('Edge Detected Image')
plt.imshow(edges, cmap='gray')
plt.axis('off')
plt.show()

print(f"Edges : {edges}")
