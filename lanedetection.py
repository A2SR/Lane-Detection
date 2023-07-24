import cv2
import numpy as np

MIN_LINE_LENGTH = 250


class Image:
    def __init__(self):
        self.mask = " "
        self.lanes = " "


def detect_lanes_hough():

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 50, 150)

    # ROI Mask
    height, width = edges.shape
    roi_mask = np.zeros_like(edges)
    roi_vertices = np.array([[(300, height), (width // 2, height // 2 + 250),
                             (width // 2 + 100, height // 2 + 250), (width - 300, height)]], dtype=np.int32)
    cv2.fillPoly(roi_mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, roi_mask)
    roi_color = (0, 255, 0)  # Green color for the ROI mask
    mask = cv2.polylines(
        image.copy(), [roi_vertices], isClosed=True, color=roi_color, thickness=2
    )

    # Detect lines
    lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi /
                            180, threshold=50, minLineLength=MIN_LINE_LENGTH, maxLineGap=200)

    # Filter lines
    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1)

        # Filter based on slope and length
        min_slope_threshold = 0.3
        min_length_threshold = 450
        if abs(slope) > min_slope_threshold and np.sqrt((x2 - x1)**2 + (y2 - y1)**2) > min_length_threshold:
            filtered_lines.append(line)

    lane_lines_image = np.zeros_like(image)

    # Draw Lane Lines
    for line in filtered_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(lane_lines_image, (x1, y1), (x2, y2), (0, 255, 0), 5)

    # Combine the original image with the detected lane lines
    lanes = cv2.addWeighted(image, 0.8, lane_lines_image, 1, 0)

    Display = Image()
    Display.mask = mask
    Display.image = lanes

    return Display


image = cv2.imread(
    "/Users/aaronrinehart/Downloads/shutterstock_1279312912-scaled.jpg")

result = detect_lanes_hough()

# Display the mask and the lane line image
cv2.imshow("Original Image with ROI Mask", result.mask)
cv2.waitKey(0)
cv2.imshow("Lane Detection Result", result.image)


cv2.waitKey(0)
cv2.destroyAllWindows()
