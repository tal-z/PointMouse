import warnings
import threading
import time
import os

import cv2
import mediapipe as mp
import pyautogui
import joblib
import numpy as np

warnings.filterwarnings(
    "ignore", category=UserWarning, module="google.protobuf.symbol_database"
)
pyautogui.FAILSAFE = False


parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class VirtualMouse:
    def __init__(
        self,
        smoothing_factor=0.1,
        frame_skip=1,
        model_path=f"{parent_dir}/models/click_model.pkl",
    ):
        # Load the trained model
        self.click_model = joblib.load(model_path)

        # Initialize MediaPipe Hands module
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.3
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Webcam initialization
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Error: Could not open webcam.")
        # Lower camera resolution for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Lower resolution
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        # Screen resolution for mapping hand movements
        self.screen_width, self.screen_height = pyautogui.size()

        # Initialize previous finger coordinates for smoothing
        self.prev_screen_x = 0
        self.prev_screen_y = 0

        # Smoothing factor (0 to 1) - smaller is smoother, larger is faster
        self.smoothing_factor = smoothing_factor

        # Frame capture control
        self.frame_lock = threading.Lock()
        self.frame = None
        self.stop_flag = False

        # Dynamic frame skipping
        self.frame_skip = frame_skip
        self.frame_count = 0

        # Throttle mouse movements to avoid excessive updates
        self.last_move_time = time.time()
        self.mouse_move_interval = 0.01  # Limit mouse updates to 100 times/second

        # Thread for frame capture
        self.capture_thread = threading.Thread(target=self.update_frame, daemon=True)
        self.capture_thread.start()

        # Left click detection
        self.left_button_pressed = False
        self.left_button_pressed_at = None
        self.click_threshold_duration = 0.3  # Duration for a click

    def update_frame(self):
        """Capture frames from the webcam in a separate thread."""
        while not self.stop_flag:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                self.stop_flag = True
                break

            # Flip the frame horizontally for a natural interaction
            frame = cv2.flip(frame, 1)

            with self.frame_lock:
                self.frame = frame.copy()

            # Small sleep to prevent high CPU usage in this thread
            time.sleep(0.01)

    def map_to_screen(self, finger_x, finger_y, cam_width, cam_height):
        """Map webcam coordinates to screen coordinates, with scaling factor to reduce hand movement range."""

        # exaggerate finger positions to reduce range of motion needed
        cam_width_center = cam_width / 2
        finger_x_offset_from_center = finger_x - cam_width_center
        finger_x += (finger_x_offset_from_center * 1.5)

        cam_height_center = cam_height / 2
        finger_y_offset_from_center = finger_y - cam_height_center
        finger_y += (finger_y_offset_from_center * 1.5)

        screen_x = min(
            int(finger_x * self.screen_width / (cam_width)),
            self.screen_width,
        )
        screen_y = max(
            min(
                int(finger_y * self.screen_height / (cam_height)),
                self.screen_height,
            ),
            0,
        )
        return screen_x, screen_y

    def smooth_movement(self, screen_x, screen_y):
        """Apply dynamic smoothing to mouse movements."""
        distance = (
            (screen_x - self.prev_screen_x) ** 2 + (screen_y - self.prev_screen_y) ** 2
        ) ** 0.5
        dynamic_smoothing = min(
            0.9, max(0.05, 0.1 + distance / 1000)
        )  # Dynamic smoothing based on movement
        smooth_x = int(
            self.prev_screen_x * (1 - dynamic_smoothing) + screen_x * dynamic_smoothing
        )
        smooth_y = int(
            self.prev_screen_y * (1 - dynamic_smoothing) + screen_y * dynamic_smoothing
        )
        # Update previous coordinates for the next frame
        self.prev_screen_x, self.prev_screen_y = smooth_x, smooth_y
        return smooth_x, smooth_y

    def detect_gesture(self, hand_landmarks):
        """Detect left and right mouse clicks."""
        if self.detect_left_click(hand_landmarks):
            return
        if self.detect_right_click(hand_landmarks):
            return
        self.detect_scroll(hand_landmarks)

    def left_click(self):
        print("left click")
        pyautogui.click(button="left")

    def right_click(self):
        print("right click")
        pyautogui.click(button="right")

    def extract_features(self, hand_landmarks):
        """Extract features from hand landmarks."""
        features = []
        for landmark in hand_landmarks.landmark:
            features.extend(
                [landmark.x, landmark.y, landmark.z]
            )  # Add x, y, z for each landmark
        return np.array(features).reshape(1, -1)  # Reshape for model input

    def process_frame(self):
        """Process the frame for hand detection and mouse control."""
        with self.frame_lock:
            frame = self.frame.copy() if self.frame is not None else None

        if frame is None:
            return

        # Convert the frame to RGB for MediaPipe processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        # Get the frame dimensions
        h, w, _ = frame.shape

        # Process hand landmarks if detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )

                # Extract features for model prediction
                features = self.extract_features(hand_landmarks)

                # Make prediction
                prediction = self.click_model.predict(features)

                # Map predicted action to mouse action
                if prediction == 0:
                    self.left_click()
                elif prediction == 1:
                    print("no click")
                elif prediction == 2:
                    self.right_click()

                # Extract forefinger mcp coordinates
                forefinger_mcp = hand_landmarks.landmark[
                    self.mp_hands.HandLandmark.INDEX_FINGER_MCP
                ]
                finger_x, finger_y = int(forefinger_mcp.x * w), int(
                    forefinger_mcp.y * h
                )

                # Map to screen coordinates
                screen_x, screen_y = self.map_to_screen(
                    finger_x, finger_y, cam_width=w, cam_height=h
                )

                # Apply smoothing to mouse movement
                smooth_x, smooth_y = self.smooth_movement(screen_x, screen_y)

                # Throttle mouse movement updates
                if time.time() - self.last_move_time > self.mouse_move_interval:
                    pyautogui.moveTo(smooth_x, smooth_y)
                    self.last_move_time = time.time()

            # Display frame
            cv2.imshow("Virtual Mouse", frame)

    def run(self):
        """Main loop to process frames."""
        while not self.stop_flag:
            if self.frame_count % self.frame_skip == 0:
                self.process_frame()

            self.frame_count += 1

            # Quit on 'Esc' key press
            if cv2.waitKey(5) & 0xFF == 27:
                self.stop_flag = True
                break

        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()

    def __del__(self):
        """Ensure resources are released properly."""
        self.stop_flag = True
        if self.capture_thread.is_alive():
            self.capture_thread.join()


if __name__ == "__main__":
    try:
        virtual_mouse = VirtualMouse(smoothing_factor=0.5, frame_skip=1)
        virtual_mouse.run()
    except Exception as e:
        print(f"An error occurred: {e}")
