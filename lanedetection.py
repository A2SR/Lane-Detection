import cv2
import numpy as np


def detect_lanes_hough(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray_image, 50, 150)

    # Apply Hough Transform to detect lines
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi /
                            180, threshold=50, minLineLength=100, maxLineGap=5)

    # Create a blank image to draw the detected lane lines
    lane_lines_image = np.zeros_like(image)

    # Draw the lane lines on the blank image
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(lane_lines_image, (x1, y1), (x2, y2), (0, 255, 0), 5)

    # Combine the original image with the detected lane lines
    result_image = cv2.addWeighted(image, 0.8, lane_lines_image, 1, 0)

    return result_image


# Read an example road image
image = cv2.imread(
    "/Users/aaronrinehart/Downloads/shutterstock_1279312912-scaled.jpg")

# Call the lane detection function
result = detect_lanes_hough(image)

# Display the original and result images
cv2.imshow("Original Image", image)
cv2.imshow("Lane Detection Result", result)

# Wait for a key press and close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
