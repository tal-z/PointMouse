import threading
import time

import cv2
import mediapipe as mp
import pyautogui


class VirtualMouse:
    def __init__(self, smoothing_factor=0.2, frame_skip=2):
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
        self.mouse_move_interval = 0.02  # Limit mouse updates to 50 times/second

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

    def detect_left_click(self, hand_landmarks, threshold=0.05):
        """Detect left-click gesture using thumb and forefinger proximity."""
        forefinger_tip = hand_landmarks.landmark[
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP
        ]
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        dist = self.calculate_distance(forefinger_tip, thumb_tip)
        if dist < threshold and not self.left_button_pressed:
            pyautogui.mouseDown(button="left")
            self.left_button_pressed_at = time.time()
            self.left_button_pressed = True
            return True
        elif dist >= threshold and self.left_button_pressed:
            pyautogui.mouseUp(button="left")
            now = time.time()
            if now - self.left_button_pressed_at < self.click_threshold_duration:
                pyautogui.click(button="left")
            self.left_button_pressed = False
            self.left_button_pressed_at = None
            return True
        return False

    def detect_right_click(self, hand_landmarks, threshold=0.025):
        """Detect right-click gesture using thumb and middle finger proximity."""
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        middle_finger_tip = hand_landmarks.landmark[
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP
        ]
        dist = self.calculate_distance(thumb_tip, middle_finger_tip)

        if dist < threshold:
            pyautogui.click(button="right")
            return True
        return False

    def detect_scroll(self, hand_landmarks, swipe_threshold=0.05, max_scroll_speed=50):
        """Detect swipe up/down gesture with the index finger for scrolling, with dynamic scroll speed."""
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]

        middle_tip = hand_landmarks.landmark[
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP
        ]
        middle_pip = hand_landmarks.landmark[
            self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP
        ]

        # Calculate vertical distance between the index finger's tip and dip
        index_vertical_distance = abs(index_tip.y - index_pip.y)
        middle_vertical_distance = abs(middle_tip.y - middle_pip.y)
        print(index_vertical_distance)

        dist_between_tips = abs(index_tip.x - middle_tip.x)

        # Check if both index and middle fingers are moving consistently in a vertical direction
        if (
            index_vertical_distance > swipe_threshold
            and middle_vertical_distance > swipe_threshold
            and dist_between_tips <= 0.03
        ):
            scroll_speed = min(int(index_vertical_distance * 100), max_scroll_speed)

            if index_tip.y < index_dip.y:  # Swipe up
                pyautogui.scroll(scroll_speed)
            else:  # Swipe down
                pyautogui.scroll(-scroll_speed)

            return True  # Scroll detected
        return False  # No scroll detected

    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points."""
        return ((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2) ** 0.5

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

                # Detect click gestures
                self.detect_gesture(hand_landmarks)

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
        virtual_mouse = VirtualMouse(smoothing_factor=0.25, frame_skip=1)
        virtual_mouse.run()
    except Exception as e:
        print(f"An error occurred: {e}")
