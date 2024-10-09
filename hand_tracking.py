import cv2
import mediapipe as mp
import pyautogui
import threading
import time


class VirtualMouse:
    def __init__(self, smoothing_factor=0.2):
        # MediaPipe Hands module
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1)  # We focus on 1 hand
        self.mp_drawing = mp.solutions.drawing_utils

        # Webcam initialization
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Error: Could not open webcam.")

        # Screen resolution for mapping hand movements
        self.screen_width, self.screen_height = pyautogui.size()

        # Initialize previous finger coordinates for smoothing
        self.prev_screen_x = 0
        self.prev_screen_y = 0

        # Smoothing factor (0 to 1) - smaller is smoother, larger is faster
        self.smoothing_factor = smoothing_factor

        # Thread control
        self.frame_lock = threading.Lock()
        self.frame = None
        self.stop_flag = False

        # Start frame capturing thread
        self.capture_thread = threading.Thread(target=self.update_frame, daemon=True)
        self.capture_thread.start()

        # Initialize left button state
        self.left_button_pressed = False

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
                self.frame = frame

            # Small sleep to prevent high CPU usage in this thread
            time.sleep(0.01)

    def map_to_screen(self, finger_x, finger_y, cam_width, cam_height):
        """Map webcam coordinates to screen coordinates."""
        screen_x = int(finger_x * self.screen_width / cam_width)
        screen_y = int(finger_y * self.screen_height / cam_height)
        return screen_x, screen_y

    def smooth_movement(self, screen_x, screen_y):
        """Smooth mouse movement using exponential moving average (EMA)."""
        smooth_x = int(
            self.prev_screen_x * (1 - self.smoothing_factor)
            + screen_x * self.smoothing_factor
        )
        smooth_y = int(
            self.prev_screen_y * (1 - self.smoothing_factor)
            + screen_y * self.smoothing_factor
        )

        # Update previous coordinates for the next frame
        self.prev_screen_x, self.prev_screen_y = smooth_x, smooth_y

        return smooth_x, smooth_y

    def detect_left_click(self, hand_landmarks, threshold=0.05):
        """Detect if thumb and forefinger are close enough to simulate a click."""
        forefinger_tip = hand_landmarks.landmark[
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP
        ]
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]

        # Calculate the Euclidean distance between thumb and forefinger
        dist = (
            (forefinger_tip.x - thumb_tip.x) ** 2
            + (forefinger_tip.y - thumb_tip.y) ** 2
        ) ** 0.5

        if dist < threshold and not self.left_button_pressed:
            pyautogui.mouseDown(button="left")  # Press down the left mouse button
            self.left_button_pressed = True
        elif dist >= threshold and self.left_button_pressed:
            pyautogui.mouseUp(button="left")  # Release the left mouse button
            self.left_button_pressed = False

    def detect_right_click(self, hand_landmarks, threshold=0.05):
        """Detect if thumb, forefinger, and middle finger are close enough to simulate a right click."""
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        forefinger_tip = hand_landmarks.landmark[
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP
        ]
        middle_finger_tip = hand_landmarks.landmark[
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP
        ]

        # Calculate distances between the thumb, forefinger, and middle finger
        dist_thumb_forefinger = (
            (thumb_tip.x - forefinger_tip.x) ** 2
            + (thumb_tip.y - forefinger_tip.y) ** 2
        ) ** 0.5
        dist_thumb_middle = (
            (thumb_tip.x - middle_finger_tip.x) ** 2
            + (thumb_tip.y - middle_finger_tip.y) ** 2
        ) ** 0.5
        dist_forefinger_middle = (
            (forefinger_tip.x - middle_finger_tip.x) ** 2
            + (forefinger_tip.y - middle_finger_tip.y) ** 2
        ) ** 0.5

        # Check if all distances are below the threshold to simulate a right-click
        if (
            dist_thumb_forefinger < threshold
            and dist_thumb_middle < threshold
            and dist_forefinger_middle < threshold
        ):
            pyautogui.click(button="right")  # Simulate a right-click

    def process_frame(self):
        """Process the frame in the main thread for hand detection and mouse control."""
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

                # Extract the coordinates of the forefinger tip
                forefinger_tip = hand_landmarks.landmark[
                    self.mp_hands.HandLandmark.INDEX_FINGER_TIP
                ]
                finger_x, finger_y = int(forefinger_tip.x * w), int(
                    forefinger_tip.y * h
                )

                # Map to screen coordinates
                screen_x, screen_y = self.map_to_screen(
                    finger_x, finger_y, cam_width=w, cam_height=h
                )

                # Apply smoothing to the mouse movement
                smooth_x, smooth_y = self.smooth_movement(screen_x, screen_y)

                print(f"Smooth Finger Coordinates: X: {smooth_x}, Y: {smooth_y}")

                # Move the mouse to the smoothed position
                pyautogui.moveTo(smooth_x, smooth_y)

                # Check for left-click gesture
                self.detect_left_click(hand_landmarks)

                # Check for right-click gesture
                self.detect_right_click(hand_landmarks)

        # Show the frame with hand tracking
        cv2.imshow("Virtual Mouse", frame)

    def run(self):
        """Run the main loop to process frames."""
        while not self.stop_flag:
            self.process_frame()

            # Press 'Esc' to quit
            if cv2.waitKey(5) & 0xFF == 27:
                self.stop_flag = True
                break

        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()

    def __del__(self):
        """Ensure thread is stopped and resources are released."""
        self.stop_flag = True
        if self.capture_thread.is_alive():
            self.capture_thread.join()


if __name__ == "__main__":
    try:
        virtual_mouse = VirtualMouse(smoothing_factor=0.2)
        virtual_mouse.run()
    except Exception as e:
        print(f"An error occurred: {e}")
