from lib.sam2.sam2.build_sam import build_sam2
from lib.sam2.sam2.sam2_image_predictor import SAM2ImagePredictor
from utils.segment_utils import *
import torch
from PIL import Image
import numpy as np
import cv2

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")
# --- Configuration ---

drawing = False     # True if the mouse button is held down
BRUSH_COLOR = (255, 255, 255) # Color to draw with (White)
MASK_ALPHA = 0.5
BRUSH_THICKNESS = 20         # Thickness of the drawing line
last_point = (-1, -1)       # Stores the last (x, y) position

def edit_mask(event, x, y, flags, param):
    """
    This function is called for every mouse event.
    """
    global drawing, erasing, mask, last_point
    is_erasing = (flags & cv2.EVENT_FLAG_CTRLKEY)
    # Event: Left mouse button is PRESSED
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        # Store the starting point
        last_point = (x, y)
        # Draw a small dot at the start
        if is_erasing:
            cv2.circle(mask, (x, y), BRUSH_THICKNESS, 0, -1)
        else:
            cv2.circle(mask, (x, y), BRUSH_THICKNESS // 2, BRUSH_COLOR, -1)

        # Event: Mouse is MOVING
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            if is_erasing:
                # Erase as the mouse moves
                cv2.line(mask, last_point, (x, y), 0, BRUSH_THICKNESS)
            else:
                # Draw a line from the last point to the current point
                cv2.line(mask, last_point, (x, y), BRUSH_COLOR, BRUSH_THICKNESS)
                # Update the last point to the current point
            # last_point = (x, y)
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

def draw_mask_on_image(image, mask, color=BRUSH_COLOR, alpha=MASK_ALPHA):
    """Draws a semi-transparent mask on top of an image."""
    output = image.copy()
    idx = (mask > 0) # Find all pixels where the mask is active
    
    if idx.any(): # Only blend if there's something to draw
        image_pixels = output[idx]
        mask_color_pixels = np.array(color, dtype=np.uint8)
        
        blended_pixels = cv2.addWeighted(
            np.full_like(image_pixels, mask_color_pixels),
            alpha,
            image_pixels,
            1 - alpha,
            0
        )
        output[idx] = blended_pixels
    
    return output

#---------------Import Image---------------#
image = Image.open('test_output/pc.jpg')
img = cv2.imread('test_output/pc.jpg')
image = np.array(image.convert("RGB"))

#---------------predict mask---------------#
sam2_checkpoint = "lib/sam2/checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

predictor = SAM2ImagePredictor(sam2_model)
predictor.set_image(image)

input_point = np.array([[434, 543]])
input_label = np.array([1])

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)
sorted_ind = np.argsort(scores)[::-1]
masks = masks[sorted_ind]
scores = scores[sorted_ind]
logits = logits[sorted_ind]
masks.shape  # (number_of_masks) x H x W
mask = masks[0].astype(np.uint8) * 255

# show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)
#--------------------editing-mask--------------------
window_name = 'Mask Eraser Tool'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(window_name, edit_mask)

print("\n--- Mask Eraser Tool ---")
print("Click and drag to ERASE parts of the mask.")
print("Press 's' to save the new mask.")
print("Press 'q' to quit.")

# --- 4. Main Display Loop ---
while True:
    
    display_image = draw_mask_on_image(img, mask)
    frame_with_brush = display_image.copy()
    # 3. Display the final frame (with the brush)
    cv2.imshow(window_name, frame_with_brush)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite("mask_edited.png", mask)
        print("\nSuccessfully saved 'mask_edited.png'!")
cv2.destroyAllWindows()

target_height, target_width = mask.shape
background = np.ones((400, 600, 3), dtype=np.uint8)
background = background*255
tile_size = 25
for i in range(0, 400, tile_size):
    for j in range(0, 600, tile_size):
        if (i // tile_size) % 2 == (j // tile_size) % 2:
            background[i:i+tile_size, j:j+tile_size] = (192, 192, 192) # light gray
        else:
            background[i:i+tile_size, j:j+tile_size] = (128, 128, 128) # dark gray

background = cv2.resize(background, (target_width, target_height))
# Step 4a: Create an inverse of the mask
# The inverse mask is a "hole" where the object should be.
mask_inv = cv2.bitwise_not(mask)

# Step 4b: "Cut out" the foreground object
# This keeps only the parts of the original_image where the mask is white (255).
foreground = cv2.bitwise_and(img, img, mask=mask)

# Step 4c: "Cut out" the background hole
# This keeps only the parts of the background where the inverse_mask is white (255).
background_hole = cv2.bitwise_and(background, background, mask=mask_inv)

# Step 4d: Combine the foreground and the background
# Since the areas are mutually exclusive, adding them pastes
# the foreground into the background's "hole".
reprojected_image = cv2.add(foreground, background_hole)

window_name = "Extracted Object"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
# --- 5. Display the results ---
cv2.imshow(window_name, reprojected_image)

print("Press any key to close...")
key = cv2.waitKey(1) & 0xFF
if key == ord('q'):
    break
elif key == ord('s'):
    cv2.imwrite("mask_extracted.png", mask)
    print("\nSuccessfully saved 'mask_extracted.png'!")
cv2.destroyAllWindows()
