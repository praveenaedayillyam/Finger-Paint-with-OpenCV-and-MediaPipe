import cv2
import numpy as np
import mediapipe as mp
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Set canvas size
width, height = 1280, 720
canvas = np.ones((height, width, 3), dtype=np.uint8) * 255  # White canvas

# Colors and their positions
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # RGB: Red, Green, Blue
color_positions = [(50, 50), (150, 50), (250, 50)]  # Positions for color circles
current_color = colors[0]  # Default to red
draw_thickness = 5

# Previous point for drawing
prev_point = None

def get_finger_status(landmarks):
    """Determine which fingers are up based on landmarks."""
    tip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
    fingers_up = []
    
    # Check if each finger is up (tip lower than PIP joint in y-coordinate)
    for tip in tip_ids[1:]:  # Skip thumb
        if landmarks[tip].y < landmarks[tip - 2].y:
            fingers_up.append(tip)
    return fingers_up

def is_over_color_circle(x, y, color_pos, radius=20):
    """Check if the finger is over a color circle."""
    return ((x - color_pos[0])**2 + (y - color_pos[1])**2) <= radius**2

def get_hand_bounding_box(landmarks, img_width, img_height):
    """Get bounding box of the hand for erasing."""
    x_coords = [int(landmark.x * img_width) for landmark in landmarks]
    y_coords = [int(landmark.y * img_height) for landmark in landmarks]
    x_min, x_max = max(0, min(x_coords) - 20), min(img_width, max(x_coords) + 20)
    y_min, y_max = max(0, min(y_coords) - 20), min(img_height, max(y_coords) + 20)
    return (x_min, y_min), (x_max, y_max)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from webcam.")
        break
    
    # Resize frame to match canvas size
    frame = cv2.resize(frame, (width, height))
    frame = cv2.flip(frame, 1)  # Mirror frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    # Create a copy of the canvas for the current frame
    display = canvas.copy()
    
    # Draw color circles
    for pos, color in zip(color_positions, colors):
        cv2.circle(display, pos, 20, color, -1)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get finger tip coordinates
            middle_tip = hand_landmarks.landmark[12]  # Middle finger for color selection
            index_tip = hand_landmarks.landmark[8]    # Index finger for drawing
            x_index, y_index = int(index_tip.x * width), int(index_tip.y * height)
            x_middle, y_middle = int(middle_tip.x * width), int(middle_tip.y * height)
            
            # Get finger status
            fingers_up = get_finger_status(hand_landmarks.landmark)
            print(f"Fingers up: {fingers_up}")  # Debug print
            
            # Color selection: only middle finger up
            if len(fingers_up) == 1 and fingers_up[0] == 12:  # Middle finger (landmark 12)
                for i, pos in enumerate(color_positions):
                    if is_over_color_circle(x_middle, y_middle, pos):
                        current_color = colors[i]
                        prev_point = None  # Reset drawing
                        print(f"Selected color: {colors[i]}")  # Debug print
                        break
            
            # Draw: only index finger up
            elif len(fingers_up) == 1 and fingers_up[0] == 8:  # Index finger (landmark 8)
                color_selected = False
                for pos in color_positions:
                    if is_over_color_circle(x_index, y_index, pos):
                        color_selected = True
                        break
                if not color_selected:
                    if prev_point is not None:
                        cv2.line(canvas, prev_point, (x_index, y_index), current_color, draw_thickness)
                        print(f"Drawing from {prev_point} to {(x_index, y_index)} with color {current_color}")  # Debug print
                    prev_point = (x_index, y_index)
            
            # Erase: all fingers up (index, middle, ring, pinky)
            elif len(fingers_up) == 4 and all(f in fingers_up for f in [8, 12, 16, 20]):  # Index, Middle, Ring, Pinky
                # Get hand bounding box
                top_left, bottom_right = get_hand_bounding_box(hand_landmarks.landmark, width, height)
                cv2.rectangle(canvas, top_left, bottom_right, (255, 255, 255), -1)  # Fill with white
                print(f"Erasing hand area: {top_left} to {bottom_right}")  # Debug print
                prev_point = None  # Reset drawing point
            
            else:
                prev_point = None  # Reset when other gestures are detected
            
            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # Combine frame and canvas
    try:
        combined = cv2.addWeighted(frame, 0.5, display, 0.5, 0)
    except cv2.error as e:
        print(f"Error in cv2.addWeighted: {e}")
        break
    
    # Display the frame
    cv2.imshow('Finger Paint', combined)
    
    # Add slight delay for stability
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    time.sleep(0.01)  # 10ms delay for smoother processing

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()