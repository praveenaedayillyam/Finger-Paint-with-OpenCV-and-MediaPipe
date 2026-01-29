# Finger Paint with OpenCV and MediaPipe

- This real-time finger-painting application, named draw_opencv.py, uses OpenCV for image processing and canvas rendering and MediaPipe for hand tracking via a webcam.
- It features a white 1280x720 canvas with three RGB color circles (Red, Green, Blue) at the top for color selection.
- To run, install dependencies with pip install opencv-python mediapipe numpy, ensure a webcam is connected, and execute python draw_opencv.py.
- The webcam feed blends with the canvas for a semi-transparent display; press 'q' to quit.
-Debug prints verify gesture detection and actions. <br><br>
- ## Gestures and Their Functions:
-	Draw: Raising only the index finger draws 5-pixel-thick lines in the selected color, following the finger's movement.
-	Select Color: Raising only the middle finger and pointing at one of the RGB circles selects that color for drawing.
-	Erase: Raising all fingers (index, middle, ring, pinky) erases the full hand area with a white rectangle over the hand's bounding box.
