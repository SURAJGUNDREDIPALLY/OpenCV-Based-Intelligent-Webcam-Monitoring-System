import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Read the first frame
ret, frame1 = cap.read()

# Create a directory to store the captured frames
os.makedirs("motion_frames", exist_ok=True)

counter = 0  # Counter for captured frames

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

    motion_detected = False

    # Draw a rectangle around the contours and save frames
    for contour in contours:
        if cv2.contourArea(contour) < 1000:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        motion_detected = True

    # Display the resulting frame
    cv2.imshow("Motion Detection", frame1)

    # Save frames when motion is detected
    flag = False
    if motion_detected:
        counter += 1
        cv2.imwrite(f"motion_frames/frame_{counter}.jpg", frame1)
        print("Motion detected! Screenshot captured.")
        motion_detected = False
        flag = True

    # Update the previous frame
    frame1 = frame2

    # Break the loop if 'q' is pressedqqqq
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if not flag:
        print('Not detected!')

# Release the video capture and destroy the windows
cap.release()
cv2.destroyAllWindows()
if flag:
    # Load the screenshot image
    screenshot = cv2.imread("motion_frames/frame_1.jpg")
    # Display the screenshot
    cv2.imshow("Motion Screenshot", screenshot)
    cv2.waitKey(0)

    # Close the window
    cv2.destroyAllWindows()

    # Path to the directory containing the screenshots
    screenshot_dir = 'motion_frames'

    # Get the list of screenshots
    screenshots = os.listdir(screenshot_dir)

    # Sort the screenshots in ascending order
    screenshots.sort()

    # Create empty lists to store the timestamps and counts
    timestamps = []
    counts = []

    # Calculate the motion event frequency
    for screenshot in screenshots:
        # Extract the timestamp from the screenshot filename
        timestamp = int(screenshot.split('_')[1].split('.')[0])
        timestamps.append(timestamp)

    # Calculate the time interval between consecutive screenshots
    time_intervals = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]

    # Count the number of screenshots in each time interval
    for interval in set(time_intervals):
        count = time_intervals.count(interval)
        counts.append(count)

    # Plot the motion event frequency
    plt.plot(range(len(counts)), counts)
    plt.xlabel('Time Interval')
    plt.ylabel('Motion Event Frequency')
    plt.title('Motion Event Frequency Over Time')
    plt.show()

    # Path to the directory containing the screenshots
    screenshot_dir = 'motion_frames'

    # Get the list of screenshots
    screenshots = os.listdir(screenshot_dir)

    # Sort the screenshots in ascending order
    screenshots.sort()

    # Create empty lists to store the timestamps and durations
    timestamps = []
    durations = []

    # Calculate the motion event duration
    for screenshot in screenshots:
        # Extract the timestamp from the screenshot filename
        timestamp = int(screenshot.split('_')[1].split('.')[0])
        timestamps.append(timestamp)

    # Calculate the duration between consecutive screenshots
    for i in range(1, len(timestamps)):
        duration = timestamps[i] - timestamps[i - 1]
        durations.append(duration)

    # Plot the motion event duration
    plt.plot(range(len(durations)), durations)
    plt.xlabel('Motion Event Index')
    plt.ylabel('Motion Event Duration')
    plt.title('Motion Event Duration Over Time')
    plt.show()

    # Path to the directory containing the screenshots
    screenshot_dir = 'motion_frames'

    # Get the list of screenshots
    screenshots = os.listdir(screenshot_dir)

    # Sort the screenshots in ascending order
    screenshots.sort()

    # Load the first screenshot to get the dimensions
    first_screenshot_path = os.path.join(screenshot_dir, screenshots[0])
    first_screenshot = cv2.imread(first_screenshot_path)
    height, width, _ = first_screenshot.shape

    # Create an empty motion heatmap
    heatmap = np.zeros((height, width), dtype=np.float32)

    # Generate the motion heatmap
    for screenshot in screenshots:
        # Load the screenshot
        screenshot_path = os.path.join(screenshot_dir, screenshot)
        frame = cv2.imread(screenshot_path)

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate the absolute difference between the grayscale frame and the first screenshot
        diff = cv2.absdiff(gray, cv2.cvtColor(first_screenshot, cv2.COLOR_BGR2GRAY))

        # Threshold the difference to identify regions of motion
        _, threshold = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        # Accumulate the threshold difference to the motion heatmap
        heatmap += threshold / 255

    # Normalize the motion heatmap
    heatmap = cv2.normalize(heatmap, None, 0, 1, cv2.NORM_MINMAX)

    # Display the motion heatmap
    plt.imshow(heatmap, cmap='hot')
    plt.colorbar()
    plt.title('Motion Heatmap')
    plt.show()
print('Executed Successfully')
