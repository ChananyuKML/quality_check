# Step 1: Import necessary libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np


# Step 3: Read the uploaded image
image_path = "test_output/pc.jpg"
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
