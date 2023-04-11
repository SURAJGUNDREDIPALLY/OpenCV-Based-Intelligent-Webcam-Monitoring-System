import cv2

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Read the first frame
ret, frame1 = cap.read()

while True:
    # Read the next frame
    ret, frame2 = cap.read()

    # Convert the frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate the absolute difference between the two frames
    diff = cv2.absdiff(gray1, gray2)

    # Threshold the difference to create a binary mask
    threshold = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]

    # Dilate the binary mask to fill in holes
    threshold = cv2.dilate(threshold, None, iterations=2)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw a rectangle around the contours
    for contour in contours:
        if cv2.contourArea(contour) < 1000:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Motion Detection", frame1)

    # Update the previous frame
    frame1 = frame2

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy the windows
cap.release()
cv2.destroyAllWindows()
