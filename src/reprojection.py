import cv2
import numpy as np
import argparse

def regist_coor(img):
    points = []
    # Check if the image was loaded successfully
    if img is None:
        print("Error: Could not load image.")
    else:
        window_name = "Image Window"
        # Create a window to display the image
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        # Bind the mouse callback function to the window
        cv2.setMouseCallback('Image Window', click_event)
        # Display the image and wait for a key press
        cv2.imshow('Image Window', img)
        cv2.waitKey(0)
    
        # Destroy all OpenCV windows
        cv2.destroyAllWindows()

def click_event(event, x, y, flags, params):
    # check if the left mouse button was clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Point : ({x}, {y})")
        points.append([x, y])



parser = argparse.ArgumentParser()
parser.add_argument('--img', type=str, default='pc.jpg')
parser.add_argument('--side', type=str, default='front')
args = parser.parse_args()

# Load your image
img = cv2.imread(args.img)

points = []
# Check if the image was loaded successfully
if img is None:
    print("Error: Could not load image.")
else:
    window_name = "Image Window"
    # Create a window to display the image
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # Bind the mouse callback function to the window
    cv2.setMouseCallback('Image Window', click_event)
    # Display the image and wait for a key press
    cv2.imshow('Image Window', img)
    key = cv2.waitKey(0)
    # Exit loop if 4 points are collected or 'q' is pressed
    if len(points) == 4:
        cv2.destroyAllWindows()
print(points)
# 1. Define the 4 points on the ORIGINAL image
# You must find these points manually or using an algorithm.
# Order: top-left, top-right, bottom-right, bottom-left
pts_src = np.array([
    points[0],  # Top-left corner of the object in the image
    points[1],  # Top-right corner
    points[2], # Bottom-right corner
    points[3]    # Bottom-left corner
], dtype="float32")

# 2. Define the 4 points for the OUTPUT image
# This is your desired "frontal" view. Let's make it 500x300 pixels.
match args.side:
    case "front":
        width, height = 1500, 250
    case "side":
        width, height = 700, 250
    case "top":
        width, height = 700, 300
pts_dst = np.array([
    [0, 0],         # Top-left
    [width, 0],  # Top-right
    [width, height], # Bottom-right
    [0, height]  # Bottom-left
], dtype="float32")

# 3. Get the transformation matrix
matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)

# 4. Apply the transformation
warped_image = cv2.warpPerspective(img, matrix, (width, height))

# 5. Save or display the result
cv2.imwrite(f'{args.side}_view_output.jpg', warped_image)
print(f'save image at {args.side}_view_output.jpg')
# To display:
# cv2.imshow("Original Image", image)
# cv2.imshow("Warped Frontal View", warped_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
