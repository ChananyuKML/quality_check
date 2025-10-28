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

ERASER_RADIUS = 20  # The size of your "eraser" brush
drawing = False     # True if the mouse button is held down
BRUSH_COLOR = (255, 255, 255) # Color to draw with (White)
MASK_ALPHA = 0.5
BRUSH_THICKNESS = 20         # Thickness of the drawing line
last_point = (-1, -1)       # Stores the last (x, y) position

def draw_mask(event, x, y, flags, param):
    """
    This function is called for every mouse event.
    """
    global drawing, last_point

    # Event: Left mouse button is PRESSED
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        # Store the starting point
        last_point = (x, y)
        # Draw a small dot at the start
        cv2.circle(img, (x, y), BRUSH_THICKNESS // 2, BRUSH_COLOR, -1)

    # Event: Mouse is MOVING
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Draw a line from the last point to the current point
            cv2.line(img, last_point, (x, y), BRUSH_COLOR, BRUSH_THICKNESS)
            # Update the last point to the current point
            last_point = (x, y)

    # Event: Left mouse button is RELEASED
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

def erase_mask(event, x, y, flags, param):
    global drawing, mask

    # Event: Left mouse button is PRESSED
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        # Erase at the first click position
        cv2.circle(mask, (x, y), ERASER_RADIUS, 0, -1)

    # Event: Mouse is MOVING
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Erase as the mouse moves
            cv2.circle(mask, (x, y), ERASER_RADIUS, 0, -1)

    # Event: Left mouse button is RELEASED
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
#---------------Import Image---------------#
image = Image.open('test_output/pc.jpg')
img = cv2.imread('test_output/pc.jpg')
image = np.array(image.convert("RGB"))


#---------------predict mask---------------#
sam2_checkpoint = "lib/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
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
mask = masks[0]

# show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)
#--------------------erase-mask--------------------
print("\n--- Mask Eraser Tool ---")
print("Click and drag to ERASE parts of the mask.")
print("Press 's' to save the new mask.")
print("Press 'q' to quit.")

window_name = 'Mask Eraser Tool'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(window_name, erase_mask)
# --- 4. Main Display Loop ---
while True:
    
    display_image = draw_mask_on_image(img, mask)
    frame_with_brush = display_image.copy()
    if last_point != (-1, -1):
        cv2.circle(
            frame_with_brush,     # Draw on the temporary frame
            last_point,    # At the mouse's current position
            BRUSH_THICKNESS // 2, # Use the brush's radius
            BRUSH_COLOR,          # Use the light blue brush color
            1                     # Use a 1-pixel outline
        )
    # 3. Display the final frame (with the brush)
    cv2.imshow(window_name, frame_with_brush)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite("mask_edited.png", mask)
        print("\nSuccessfully saved 'mask_edited.png'!")
    
    cv2.destroyAllWindows()

