import cv2
import mediapipe as mp
import csv
import os

# Define the labels
LABELS = ["left click", "right click", "no click"]

# MediaPipe Hands initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7
)


# Function to save landmarks and labels to a CSV file
def save_landmarks_to_csv(label, landmarks, csv_file="training_data/hand_landmarks.csv"):
    # Flatten the landmark data and prepend the label
    data_row = [label] + [
        coord
        for landmark in landmarks
        for coord in (landmark.x, landmark.y, landmark.z)
    ]

    # Append the data to the CSV file
    with open(csv_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(data_row)
    print(f"Landmarks saved with label '{label}'.")


def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)

    # CSV file for saving landmarks and labels
    csv_file = "training_data/hand_landmarks.csv"

    # Check if CSV file exists, if not create it with a header row
    if not os.path.exists(csv_file):
        with open(csv_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            # Create header with "label" followed by landmark coordinates (21 landmarks, each with x, y, z)
            header = ["label"]
            for i in range(21):
                header.append(f"x{i}")
                header.append(f"y{i}")
                header.append(f"z{i}")

            writer.writerow(header)

    print(
        f"Recording data. Press '1' for 'left click', '2' for 'right click', '3' for 'no click'."
    )
    captured_landmarks = None

    while True:
        # Capture frame-by-frame from the webcam
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame.")
            break

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and detect hand landmarks
        result = hands.process(frame_rgb)

        # Draw the hand landmarks on the frame (for visual feedback)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

        # Display the live video feed
        cv2.imshow(
            'Live Feed - Press "1" for left click, "2" for right click, "3" for no click',
            frame,
        )

        # Listen for keypresses
        key = cv2.waitKey(1) & 0xFF

        # Press '1', '2', or '3' to label the image
        if key in [ord("1"), ord("2"), ord("3")]:
            if result.multi_hand_landmarks:
                captured_landmarks = result.multi_hand_landmarks[
                    0
                ].landmark  # Save the landmarks
                captured_frame = frame.copy()  # Save the captured frame

                if key == ord("1"):
                    label = "left click"
                elif key == ord("2"):
                    label = "right click"
                elif key == ord("3"):
                    label = "no click"

                # Save the landmarks and label
                save_landmarks_to_csv(label, captured_landmarks, csv_file)

                # Display the captured frame for visual feedback
                cv2.imshow("Captured Frame - Ready to Label", captured_frame)
                print(
                    f"Image captured with label '{label}'. Ready for the next capture."
                )

        # Press 'q' to quit the program
        if key == ord("q"):
            break

    # Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
