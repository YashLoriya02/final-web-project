import cv2
import numpy as np

# Open a video file or capture device (0 for the default camera)
cap = cv2.VideoCapture(0)  # Change to 'video.mp4' for a video file

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video stream

    # Convert to grayscale for processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Adaptive Thresholding: Produces a binary image based on local pixel neighborhood
    # Parameters: maxValue, adaptiveMethod, thresholdType, blockSize, C
    adaptive_thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # Smoothing: Apply Gaussian Blur to reduce noise and smooth the image
    # You can experiment with the kernel size (e.g., (5,5))
    smooth = cv2.GaussianBlur(frame, (5, 5), 0)

    # Edge Detection: Use Canny edge detector on the grayscale image
    edges = cv2.Canny(gray, 100, 200)

    # Bitwise Operations: Use the adaptive threshold as a mask to isolate parts of the smoothed image
    # Here we apply a bitwise 'and' between the smoothed frame and itself using the mask
    bitwise_result = cv2.bitwise_and(smooth, smooth, mask=adaptive_thresh)

    # Display the original and processed frames in separate windows
    cv2.imshow('Original', frame)
    cv2.imshow('Adaptive Threshold', adaptive_thresh)
    cv2.imshow('Smoothed', smooth)
    cv2.imshow('Edge Detection', edges)
    cv2.imshow('Bitwise Operation', bitwise_result)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
