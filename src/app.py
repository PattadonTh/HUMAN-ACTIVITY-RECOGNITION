import os
import cv2
import mediapipe as mp
import math
from collections import deque, Counter

# Disable OneDNN for TensorFlow and minimize log levels
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Buffer for gesture smoothing (Increased Buffer Size)
gesture_buffer = deque(maxlen=20)  # Buffer of last 20 detected activities

# Global variable to track previous foot position for walking detection
prev_foot_index_y = None


# Function to calculate the angle between two points
def calculate_angle(x1, y1, x2, y2):
    return math.degrees(math.atan2(y2 - y1, x2 - x1))


# Function to detect if hands are close enough for clapping
def is_clapping(pose_landmarks, threshold=0.03):
    left_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    return (
        abs(left_wrist.x - right_wrist.x) < threshold
        and abs(left_wrist.y - right_wrist.y) < threshold
    )


# Function to detect if either hand is raised
def is_raising_hand(pose_landmarks, hand="right"):
    wrist = pose_landmarks.landmark[
        (
            mp_pose.PoseLandmark.RIGHT_WRIST
            if hand == "right"
            else mp_pose.PoseLandmark.LEFT_WRIST
        )
    ].y
    shoulder = pose_landmarks.landmark[
        (
            mp_pose.PoseLandmark.RIGHT_SHOULDER
            if hand == "right"
            else mp_pose.PoseLandmark.LEFT_SHOULDER
        )
    ].y
    return wrist < shoulder


# Function to detect walking based on foot movement
def is_walking(pose_landmarks, threshold=0.0009):
    global prev_foot_index_y
    left_foot_index_y = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y
    right_foot_index_y = pose_landmarks.landmark[
        mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
    ].y
    current_position = (left_foot_index_y + right_foot_index_y) / 2

    if prev_foot_index_y is not None:
        movement = abs(current_position - prev_foot_index_y)
        prev_foot_index_y = current_position
        return movement > threshold
    prev_foot_index_y = current_position
    return False


# Function to detect activities based on pose landmarks, ignoring rotation detection
def detect_activity(pose_landmarks):
    if is_walking(pose_landmarks):
        return "Walking"
    elif is_clapping(pose_landmarks):
        return "Clapping"
    elif is_raising_hand(pose_landmarks, "right"):
        return "Raising Right Hand"
    elif is_raising_hand(pose_landmarks, "left"):
        return "Raising Left Hand"
    return "Standing"


# Function to draw a bounding rectangle around the human body with adjustable size
def draw_bounding_box(
    image, landmarks, image_width, image_height, margin=0.1, scale=1.1
):
    x_min = min([landmark.x for landmark in landmarks]) * image_width
    x_max = max([landmark.x for landmark in landmarks]) * image_width
    y_min = min([landmark.y for landmark in landmarks]) * image_height
    y_max = max([landmark.y for landmark in landmarks]) * image_height

    # Apply margin and scale
    x_min = max(0, x_min - (x_max - x_min) * margin)  # Add margin to x_min
    x_max = min(image_width, x_max + (x_max - x_min) * margin)  # Add margin to x_max
    y_min = max(0, y_min - (y_max - y_min) * margin)  # Add margin to y_min
    y_max = min(image_height, y_max + (y_max - y_min) * margin)  # Add margin to y_max

    # Apply scaling factor
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    width = (x_max - x_min) * scale
    height = (y_max - y_min) * scale
    x_min = max(0, center_x - width / 2)
    x_max = min(image_width, center_x + width / 2)
    y_min = max(0, center_y - height / 2)
    y_max = min(image_height, center_y + height / 2)

    # Draw the rectangle
    cv2.rectangle(
        image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2
    )


# Weighted smoothing function to prioritize recent gestures
def smooth_gesture_weighted(
    gesture_buffer, current_gesture, recent_weight=2, required_consistency=8
):
    if current_gesture:
        gesture_buffer.append(current_gesture)

    # Create a Counter for the gestures in the buffer
    gesture_counter = Counter(gesture_buffer)

    # Apply more weight to recent gestures
    for i in range(1, min(len(gesture_buffer), required_consistency + 1)):
        gesture_counter[gesture_buffer[-i]] += recent_weight * (
            required_consistency + 1 - i
        )

    # Get the gesture with the highest weighted count
    most_common_gesture = gesture_counter.most_common(1)[0][0]

    return most_common_gesture


if __name__ == "__main__":
    video = "../assets/video_test.mp4"
    cap = cv2.VideoCapture(video)
    current_activity = "Standing"  # Initialize with default activity

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB for MediaPipe processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            # Detect activity
            detected_activity = detect_activity(results.pose_landmarks)

            # Apply weighted gesture smoothing
            current_activity = smooth_gesture_weighted(
                gesture_buffer, detected_activity
            )

            # Get image dimensions
            image_height, image_width, _ = frame.shape

            # Draw bounding rectangle around the human body with adjustable size
            draw_bounding_box(
                frame,
                results.pose_landmarks.landmark,
                image_width,
                image_height,
                margin=0.1,
                scale=1.1,
            )

            # Display smoothed activity on the video frame
            cv2.putText(
                frame,
                current_activity,
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (255, 0, 0),
                3,
                cv2.LINE_AA,
            )

        # Resize the frame and display it
        frame = cv2.resize(frame, (1000, 800))
        cv2.imshow("Frame", frame)

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()
