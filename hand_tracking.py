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
        self.finger_together_time = 0.0  # Time fingers are together
        self.click_threshold_duration = 0.1  # Duration for a click (in seconds)

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

    def detect_click(self, hand_landmarks, threshold=0.02):
        """Detect if thumb and forefinger are close enough to simulate a click."""
        if self.detect_right_click(hand_landmarks, threshold):
            return
        self.detect_left_click(hand_landmarks, threshold)

    def detect_left_click(self, hand_landmarks, threshold=0.02):
        forefinger_tip = hand_landmarks.landmark[
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP
        ]
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]

        # Calculate the Euclidean distance between thumb and forefinger
        dist = self.calculate_distance(forefinger_tip, thumb_tip)

        # Check if fingers are together
        if dist < threshold:
            if not self.left_button_pressed:
                # Mouse down
                pyautogui.mouseDown(button="left")  # Press down the left mouse button
                # Start timing the fingers being together
                self.left_button_pressed = True
                self.finger_together_time = (
                    time.time()
                )  # Record the time they first touch

        else:
            # If fingers are apart, reset everything
            if self.left_button_pressed:
                pyautogui.mouseUp(
                    button="left"
                )  # Release the left mouse button if previously pressed
                self.left_button_pressed = False  # Reset mouse down state

                duration = duration = time.time() - self.finger_together_time
                print(duration)
                if duration < self.click_threshold_duration:
                    pyautogui.click(button="left")  # Simulate a click (press + release)

            self.finger_together_time = 0.0  # Reset the timer

    def detect_right_click(self, hand_landmarks, threshold=0.02):
        """Detect if thumb, forefinger, and middle finger are close enough to simulate a right-click."""
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        middle_finger_tip = hand_landmarks.landmark[
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP
        ]

        # Calculate distances between fingers
        dist_thumb_middle = self.calculate_distance(thumb_tip, middle_finger_tip)

        # Check if all distances are below the threshold to simulate a right-click
        if dist_thumb_middle < threshold:
            pyautogui.click(button="right")  # Simulate a right-click
            return True
        return False

    def calculate_distance(self, point1, point2):
        """Calculate the Euclidean distance between two points."""
        return ((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2) ** 0.5

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
                self.detect_click(hand_landmarks)

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
