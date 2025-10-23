import cv2
import numpy as np

def click_event(event, x, y, flags, params):
    # Check if the left mouse button was clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Coordinates: ({x}, {y})")

def regist_coor(img):
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
# Load your image
image = cv2.imread('chess_board.jpg')

regist_coor(image)
# 1. Define the 4 points on the ORIGINAL image
# You must find these points manually or using an algorithm.
# Order: top-left, top-right, bottom-right, bottom-left
pts_src = np.array([
    [511, 380],  # Top-left corner of the object in the image
    [1403, 483],  # Top-right corner
    [1279, 1123],  # Bottom-right corner
    [496, 1018]    # Bottom-left corner
], dtype="float32")

# 2. Define the 4 points for the OUTPUT image
# This is your desired "frontal" view. Let's make it 500x300 pixels.
width, height = 500, 500
pts_dst = np.array([
    [0, 0],         # Top-left
    [width, 0],  # Top-right
    [width, height], # Bottom-right
    [0, height]  # Bottom-left
], dtype="float32")

# 3. Get the transformation matrix
matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)

# 4. Apply the transformation
warped_image = cv2.warpPerspective(image, matrix, (width, height))

# 5. Save or display the result
cv2.imwrite('frontal_view_output.jpg', warped_image)
print('save image at frontal_view_output.jpg')
# To display:
# cv2.imshow("Original Image", image)
# cv2.imshow("Warped Frontal View", warped_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
