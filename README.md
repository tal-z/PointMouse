# PointMouse!

PointMouse lets you use your hand to intuitively control your computer mouse. It uses your webcam to detect your hand's position, and calculates where on the screen you are pointing.


# Data Collection Script for ML Model Training
The `collect_training_data.py` script utilizes the MediaPipe library and OpenCV to recognize hand positions and save corresponding landmark data to a CSV file with a user-applied label. The program works as follows:

1) Captures live video feed from a webcam, detecting hand landmarks
2) Allows the user to grab a frame with hand landmarks
3) Apply a label to the landmarks
4) Write the hand landmarks and corresponding label to a new row in a CSV file

To run the script: `python collect_training_data.py`

#### Controls

1) When you start the program, it will open a live video feed from your webcam.
2) Press `r` to capture an image from the feed.
3) After capturing, you will see the captured frame in a separate window. Press:
    - 1 for "left click" label
    - 2 for "right click" label
    - 3 for "no click" label

4) Press `q` to quit the program.

#### Output & Data Format
The landmarks and their associated labels will be saved in a CSV file named hand_landmarks.csv in the same directory as the script.

###### Data Format 
The CSV file hand_landmarks.csv will contain the following structure:

    Header:
        label: The gesture label ("left click", "right click", "no click")
        x0, y0, z0, x1, y1, z1, ..., x20, y20, z20: The x, y, z coordinates for each of the 21 hand landmarks.

    Example Row:
        left click, 0.123, 0.456, 0.789, ..., 0.111, 0.222, 0.333


#### Hand Landmarks
![Hand Landmarks](hand-landmarks.png)



Interesting problems:
 - When the hand goes off camera, or even hand landmarks go off camera, accuracy goes way down
    - Solution: keep hand on screen by reducing necessary range of motion to move cursor