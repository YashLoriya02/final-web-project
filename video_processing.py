import cv2
import numpy as np

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    adaptive_thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    smooth = cv2.GaussianBlur(frame, (5, 5), 0)

    edges = cv2.Canny(gray, 100, 200)

    bitwise_result = cv2.bitwise_and(smooth, smooth, mask=adaptive_thresh)

    cv2.imshow('Original', frame)
    cv2.imshow('Adaptive Threshold', adaptive_thresh)
    cv2.imshow('Smoothed', smooth)
    cv2.imshow('Edge Detection', edges)
    cv2.imshow('Bitwise Operation', bitwise_result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
