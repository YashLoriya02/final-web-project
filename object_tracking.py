import cv2

# Initialize the video capture (0 for the default camera or provide a video file name)
cap = cv2.VideoCapture(0)  # Change to 'video.mp4' to use a video file

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

# Read the first frame of the video
ret, frame = cap.read()
if not ret:
    print("Error: Cannot read video frame.")
    exit()

# Let the user select the object to track by drawing a bounding box
roi = cv2.selectROI("Select Object to Track", frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Select Object to Track")

# Create the tracker. Options include: MIL, KCF, TLD, MEDIANFLOW, GOTURN, MOSSE, CSRT
# Here, we use CSRT for its robustness
tracker = cv2.TrackerCSRT_create()

# Initialize the tracker with the first frame and the selected bounding box
tracker.init(frame, roi)

while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video stream

    # Update the tracker and get the updated position
    success, bbox = tracker.update(frame)

    if success:
        # Draw a bounding box around the tracked object
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2, 1)
        cv2.putText(frame, "Tracking", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    else:
        # Tracking failure detected
        cv2.putText(frame, "Tracking failure detected", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Object Tracking", frame)

    # Exit if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
