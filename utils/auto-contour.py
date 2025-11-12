import cv2
import numpy as np

def get_four_corners(image_path):
    """
    Finds the four corners of the largest contour in the image.
    Returns the points in order: top-left, top-right, bottom-right, bottom-left
    """
    
    # --- 1. Load and Pre-process the Image ---
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None, None

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply thresholding. This is a critical step!
    # cv2.THRESH_OTSU automatically finds the best threshold value.
    # We get a binary image: black background, white object.
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # --- 2. Find Contours ---
    # Find all outlines in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("No contours found.")
        return None, image # Return original image for display

    # --- 3. Find the Largest Contour ---
    # We assume the largest contour is our product
    largest_contour = max(contours, key=cv2.contourArea)
    
    # --- 4. Approximate the Contour Shape ---
    # Get the perimeter of the contour
    perimeter = cv2.arcLength(largest_contour, True)
    
    # Approximate the contour to a simpler shape (e.g., a quadrilateral)
    # The 0.02 * perimeter is the "epsilon" value - it controls the
    # precision of the approximation. You might need to tune this.
    approx_corners = cv2.approxPolyDP(largest_contour, 0.02 * perimeter, True)
    
    # --- 5. Get the Four Corners ---
    # Check if our approximation has 4 points
    if len(approx_corners) == 4:
        # approx_corners is in the shape [[p1], [p2], [p3], [p4]]
        # Squeeze it to be [[x1, y1], [x2, y2], ...]
        points = approx_corners.reshape(4, 2)
        print("Found 4 corners:", points)
        
        # Draw the corners on the original image
        viz_image = image.copy()
        for (x, y) in points:
            cv2.circle(viz_image, (x, y), 10, (0, 0, 255), -1) # Red dots
            
        # Reorder the points to be [top-left, top-right, bottom-right, bottom-left]
        # This is necessary for cv2.getPerspectiveTransform
        ordered_points = order_points(points)
        return ordered_points, viz_image
        
    else:
        print(f"Found {len(approx_corners)} points, not 4. Try adjusting the epsilon value.")
        return None, image # Return original image for display

def order_points(pts):
    """
    Sorts the 4 points into a consistent order:
    top-left, top-right, bottom-right, bottom-left
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # Top-left has smallest sum (x+y), bottom-right has largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Top-right has smallest diff (y-x), bottom-left has largest diff
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

# --- Main execution ---
image_file = 'img/pc.jpg'
src_points, debug_image = get_four_corners(image_file)

if src_points is not None:
    print("Ordered Source Points:\n", src_points)
    
    # Now you can use these src_points for your perspective transform!
    # (Code from previous answer)
    
    # 1. Define Destination Points
    output_width = 600
    output_height = 400
    dst_points = np.float32([
        [0, 0],
        [output_width, 0],
        [output_width, output_height],
        [0, output_height]
    ])

    # 2. Calculate and Apply Transform
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    img_reprojected = cv2.warpPerspective(debug_image, M, (output_width, output_height))

    cv2.imshow("1. Image with Corners", debug_image)
    cv2.imshow("2. Reprojected (Flat) Image", img_reprojected)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Could not find 4 corners.")
    cv2.imshow("Failed Detection", debug_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()