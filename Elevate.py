import cv2
from flask import Flask, render_template, Response, jsonify, request, session
import mediapipe as mp
import numpy as np
import warnings
import cv2
import numpy as np
import time
import mediapipe as mp
from scipy.spatial import distance as dist
from imutils import face_utils
import dlib
from keras.models import model_from_json
import math as m
from flask_wtf.csrf import CSRFProtect
from datetime import datetime, timezone
import pickle
import pandas as pd
import os


warnings.filterwarnings(action="ignore", category=UserWarning)

app = Flask(__name__)
app.config['SECRET_KEY'] = '\xfeP\xe5\x16\x8b\xa8P\xe1\xaaJ D8\x0e\x1f\x98'
csrf = CSRFProtect(app)


from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///Elevate.db'
db = SQLAlchemy(app)
migrate = Migrate(app, db)

from flask_login import LoginManager, login_user, login_required, logout_user, current_user, UserMixin

# After db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(100), nullable=False)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)

    # Add a method to set password hash
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    # Add a method to check password
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)



# Initilize medipipe selfie segmentation class.
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2-x1)**2+(y2-y1)**2)
    return dist

def findDistance2(x1, y1, x2, y2):
    # Calculate Euclidean distance
    dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    # Determine sign based on the position of the wrists
    sign = np.sign(x2 - x1)
    return dist, sign

# Calculate angle.
def findAngle(x1, y1, x2, y2):
    theta = m.acos((y2 -y1)*(-y1) / (m.sqrt((x2 - x1)**2 + (y2 - y1)**2) * y1))
    degree = int(180/m.pi)*theta
    return degree

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle

def sendWarning():
    print('YOU ARE IN WRONG POSITION FOR LONG TIME')

#show display
show_display = True

#------------------ Posture Setup -----------
good_posture_start_time = None
bad_posture_start_time = None
last_known_posture = None
is_bad_posture_detected = False #sends alert if posture bad for too long


posture_detection = []

# Initilize frame counters.
good_frames = 0
bad_frames = 0

# Font type.
font = cv2.FONT_HERSHEY_SIMPLEX

# Colors.
blue = (255, 127, 0)     
red = (50, 50, 255)
green = (127, 255, 0)
dark_blue = (127, 20, 0)
light_green = (127, 233, 100)
yellow = (0, 255, 255)
pink = (255, 0, 255)

# Initialize mediapipe pose class.
mp_pose = mp.solutions.pose 
pose = mp_pose.Pose() #calls the Pose function via the mp_pose variable 

# Initialize video capture object


webcam = cv2.VideoCapture(0)
fps = int(webcam.get(cv2.CAP_PROP_FPS))
width = int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Initialize video writer.
video_output = cv2.VideoWriter('/content/output.mp4', fourcc, fps, frame_size)



mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# ---------------------------- nodding logic declaration ----------------------------------

previous_nose_y = None
nod_sequence = []  # Tracks the sequence of movements
nod_detected_frames = 0
sequence_start_time = None
MAX_NOD_DURATION = 1.0  # Maximum duration in seconds for a nod


# Initialize variables for yawning detection
yawning_start_time = None
yawning_threshold = 27  # Lip distance threshold for yawning
yawning_duration = 3    # Duration in seconds to confirm yawning
surprise_threshold = 14

# # Load the emotion detection model
# json_file = open("D:/Projects vs code/Elevate/Face_Detection/emotiondetector.json", "r")
# model_json = json_file.read()
# json_file.close()
# model = model_from_json(model_json)
# model.load_weights("D:/Projects vs code/Elevate/Face_Detection/emotiondetector.h5")

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build the path to the JSON file
json_path = os.path.join(script_dir, 'Face_Detection', 'emotiondetector.json')

# Build the path to the weights file
weights_path = os.path.join(script_dir, 'Face_Detection', 'emotiondetector.h5')

# Load the model JSON file
with open(json_path, 'r') as json_file:
    model_json = json_file.read()

# Load the model from JSON file
model = model_from_json(model_json)

# Load the weights into the model
model.load_weights(weights_path)

# Initialize Haar Cascade, MediaPipe, and dlib
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
hands = mp_hands.Hands(max_num_hands=2,min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5)
detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("Face_Detection/shape_predictor_68_face_landmarks.dat")
script_dir = os.path.dirname(os.path.abspath(__file__))
# Build the path to the '.dat' file
dat_path = os.path.join(script_dir, 'Face_Detection', 'shape_predictor_68_face_landmarks.dat')
# Use the path with dlib
predictor = dlib.shape_predictor(dat_path)
mp_drawing = mp.solutions.drawing_utils


# ------------------ face touch setup ------------------------------

is_touching_face = False
face_touch_start_time = None
is_touching_face_long = False

# --------------- Blink detection setup ==============================

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3
COUNTER = 0
TOTAL = 0
total_blinks_accumulated = 0 
intervals_passed = 0
start_time = time.time()
blink_data = []

last_posture_change_time = time.time()

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


# --------------------- Emotion Detection Setup ------------------------------


def extract_features(frame):
    feature = np.array(frame)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

def get_lip_distance(landmarks, frame_width, frame_height):
    def to_tuple(landmark):
        return (int(landmark.x * frame_width), int(landmark.y * frame_height))

    upper_lip_indices = [81, 82, 13, 312]
    lower_lip_indices = [178, 87, 14, 317]
    upper_lip_point = np.mean([to_tuple(landmarks[idx]) for idx in upper_lip_indices], axis=0).astype(int)
    lower_lip_point = np.mean([to_tuple(landmarks[idx]) for idx in lower_lip_indices], axis=0).astype(int)
    distance = np.linalg.norm(upper_lip_point - lower_lip_point)
    return distance, upper_lip_point, lower_lip_point

def draw_landmarks(frame, landmarks):
    for landmark in landmarks:
        x = int(landmark.x * frame.shape[1])
        y = int(landmark.y * frame.shape[0])
        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)


#pose setup

last_known_crossed_arms = None
crossed_arms_start_time = None

last_known_hands_behind_head = None
hands_behind_head_start_time = None

pose_durations = []

#surprise setup
        
surprise_counter = 0
last_surprise_time = None
surprise_cooldown = 2   

#hand gesture detection
is_gesture_detected = False
gesture_start_time = None
last_gesture_class = None

# Webcam initialization and emotion labels

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

emotion_start_time = {}  # Dictionary to track the start time for each emotion
emotion_durations = {}  # Dictionary to track the duration of each emotion

last_known_emotion = None

speech_results_sorted = []

is_running = True
combined_results = []
combined_results_sorted = []
detections = []
emotion_detection = []


def gen():


    """Video streaming generator function."""
    global good_frames, bad_frames, COUNTER, TOTAL, EYE_AR_THRESH, EYE_AR_CONSEC_FRAMES, is_running, combined_results, combined_results_sorted, previous_nose_y, nod_sequence, nod_detected_frames,sequence_start_time,MAX_NOD_DURATION, total_blinks_accumulated, intervals_passed, start_time
    global good_posture_start_time, bad_posture_start_time, last_known_posture, last_posture_change_time
    global is_touching_face, face_touch_start_time 
    global is_gesture_detected, gesture_start_time, last_gesture_class     
    last_known_emotion = None
    global surprise_counter, last_surprise_time, surprise_cooldown 
    global emotion_detection
    global posture_detection
    global last_known_crossed_arms,crossed_arms_start_time,last_known_hands_behind_head,hands_behind_head_start_time 
    global pose_durations
    global intervals_passed
    global is_bad_posture_detected
    global is_touching_face_long
    global show_display
    
    if not webcam.isOpened():
        detections.append((time.time(), "Error: Camera is not accessible"))
        return

    while is_running:
        ret, frame = webcam.read()  
        if not ret:
            break
        

        neck_inclination = 0
        torso_inclination = 0
        l_shldr_x = l_shldr_y = r_shldr_x = r_shldr_y = None
        prediction_label = "" 
        #posture detection
        # Get fps, height, and width.
        fps = webcam.get(cv2.CAP_PROP_FPS)
        h, w = frame.shape[:2]

        # Convert the BGR frame to RGB and process it.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        keypoints = pose.process(rgb_frame)

        lm = keypoints.pose_landmarks
        lmPose = mp_pose.PoseLandmark


        # Ensure keypoints are detected.
        if keypoints.pose_landmarks:
            # Acquire the landmark coordinates and calculate angles, positions, etc.
            # Once aligned properly, left or right should not be a concern.      
            # Left shoulder.
            l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
            l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
            r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
            r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)


            # Left ear.
            l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
            l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)

            #Right ear
            r_ear_x = int(lm.landmark[lmPose.RIGHT_EAR].x * w)
            r_ear_y = int(lm.landmark[lmPose.RIGHT_EAR].y * h)


            # Left hip.
            l_nose_x = int(lm.landmark[lmPose.NOSE].x * w)
            l_nose_y = int(lm.landmark[lmPose.NOSE].y * h)


            right_wrist_x = int(lm.landmark[lmPose.RIGHT_WRIST].x * w)
            right_wrist_y = int(lm.landmark[lmPose.RIGHT_WRIST].y * h)

            left_wrist_x = int(lm.landmark[lmPose.LEFT_WRIST].x * w)
            left_wrist_y = int(lm.landmark[lmPose.LEFT_WRIST].y * h)

            right_elbow_x = int(lm.landmark[lmPose.RIGHT_ELBOW].x * w)
            right_elbow_y = int(lm.landmark[lmPose.RIGHT_ELBOW].y * h)

            left_elbow_x = int(lm.landmark[lmPose.LEFT_ELBOW].x * w)
            left_elbow_y = int(lm.landmark[lmPose.LEFT_ELBOW].y * h)

            nose_landmark = keypoints.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
            nose_y = int(nose_landmark.y * h)
                

            right_wrist = (right_wrist_x, right_wrist_y)
            left_wrist = (left_wrist_x, left_wrist_y)

            right_elbow = (right_elbow_x, right_elbow_y)
            left_elbow = (left_elbow_x, left_elbow_y)

            right_shoulder = (r_shldr_x, r_shldr_y)
            left_shoulder = (l_shldr_x, l_shldr_y)

            left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

                
            wrist_dist, sign = findDistance2(right_wrist_x,right_wrist_y,left_wrist_x, left_wrist_y)
            ear_wrist_dist_r = findDistance2(right_wrist_x,right_wrist_y,r_ear_x,r_ear_y)
            ear_wrist_dist_l = findDistance2(left_wrist_x,left_wrist_y,l_ear_x,l_ear_y)

            # Calculate angles.
            neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
            torso_inclination = findAngle(l_nose_x, l_nose_y, l_shldr_x, l_shldr_y)


            #Logic for pose estimation
            mid_x = int((right_wrist_x + left_wrist_x) / 2)
            mid_y = int((right_wrist_y + left_wrist_y) / 2)

            display_dist = -wrist_dist if sign < 0 else wrist_dist


            if previous_nose_y is not None:
                movement_threshold = 2  # Threshold for detecting significant movement
                y_diff = nose_y - previous_nose_y

                current_time = time.time()
                if y_diff < -movement_threshold:  # Moved up
                    if not nod_sequence:
                        sequence_start_time = current_time
                    nod_sequence.append(('up', current_time))
                    
                elif y_diff > movement_threshold:  # Moved down
                    if not nod_sequence:
                        sequence_start_time = current_time
                    nod_sequence.append(('down', current_time))
                    

                # Check if sequence is too slow
                if sequence_start_time and (current_time - sequence_start_time > MAX_NOD_DURATION):
                    nod_sequence.clear()

                # Check for a complete nod sequence: 'down' then 'up' or 'up' then 'down'
                if len(nod_sequence) >= 2 and ((nod_sequence[-2][0] != nod_sequence[-1][0]) and
                                            (nod_sequence[-1][1] - nod_sequence[-2][1] <= MAX_NOD_DURATION)):
                    nod_sequence.clear()  # Reset the sequence after detecting a nod
                    # cv2.putText(frame, "Nod Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  
                    detections.append((time.time(), "User is Nodding"))
                    # Message to display
                    if show_display :
                        nod_message = "Nod Detected"

                        # Calculate the text size to determine the box size
                        (text_width, text_height), _ = cv2.getTextSize(nod_message, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

                        # Frame dimensions
                        frame_height, frame_width = frame.shape[:2]

                        # Define the box's coordinates based on the text size and frame width
                        # Align the box to the top right corner with some padding from the edges
                        padding = 10  # Padding from the edges of the frame
                        box_top_right = (frame_width - text_width - 2 * padding, padding)
                        box_bottom_left = (frame_width - padding, text_height + 2 * padding)

                        # Draw a semi-transparent rectangle as the background
                        overlay = frame.copy()
                        cv2.rectangle(overlay, box_top_right, box_bottom_left, (0, 255, 0), cv2.FILLED)  # Green box for nod detection
                        alpha = 0.4  # Transparency factor
                        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                        # Adjust text position to be inside the box, aligning it properly
                        text_position = (box_top_right[0] + padding, box_top_right[1] + text_height + padding - 5)  # Adjust text position within the box

                        # Draw the text on top of the rectangle
                        cv2.putText(frame, nod_message, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # White text for readability


            previous_nose_y = nose_y

            


            # Put text on frame
            # cv2.putText(frame, f"{display_dist:.2f}", (mid_x, mid_y - 10), font, 0.9, light_green, 2)
            
        
            # cv2.line(frame, left_shoulder, left_elbow, green, 4)
            # cv2.line(frame, left_elbow, left_wrist, green, 4)
            # cv2.line(frame, right_shoulder, right_elbow, green, 4)
            # cv2.line(frame, right_elbow, right_wrist, green, 4)

            # Display angles

            # cv2.putText(frame, f"L: {left_elbow_angle:.2f}°", (left_elbow[0] - 50, left_elbow[1] - 10), font, 0.5, green, 2)     
            # cv2.putText(frame, f"R: {right_elbow_angle:.2f}°", (right_elbow[0] + 10, right_elbow[1] - 10), font, 0.5, light_green, 2)

            ear_wrist_dist_r_distance, _ = ear_wrist_dist_r
            ear_wrist_dist_l_distance, _ = ear_wrist_dist_l

            # Use the extracted distance value for the formatted string
            test_text_r = f"R Ear-Wrist Dist: {ear_wrist_dist_r_distance:.2f}"
            test_text_l = f"L Ear-Wrist Dist: {ear_wrist_dist_l_distance:.2f}"

            # Assuming you have correctly defined positions for text display
            position_right = (10, h - 40)  # Adjust as needed for right ear-wrist distance display
            position_left = (10, h - 10)  # Adjust as needed for left ear-wrist distance display

            # Display the distances using cv2.putText
            # cv2.putText(frame, test_text_r, position_right, font, 0.5, light_green, 2)
            # cv2.putText(frame, test_text_l, position_left, font, 0.5, light_green, 2)

            # cv2.putText(frame, f"L Ear-Wrist Dist: {ear_wrist_dist_l:.2f}", text_position_left_ear_wrist, font, 0.5, colors["light_green"], 2)

            current_time = time.time()

            frame_height, frame_width = frame.shape[:2]

            # Common styling
            text_size = 1
            text_thickness = 2
            background_color = (100, 100, 255)  # Soft shade of blue
            text_color = (255, 255, 255)  # White text
            alpha = 0.4  # Transparency factor for the rectangle

            # Crossed Arms Detection
            crossed_arms_condition = 55 <= left_elbow_angle <= 100 and 55 <= right_elbow_angle <= 100 and sign < 30
            if crossed_arms_condition:
                if last_known_crossed_arms is None:  # Pose just started
                    last_known_crossed_arms = "Crossed Arms"
                    crossed_arms_start_time = current_time
                # cv2.putText(frame, "Crossed Arms Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                if show_display:
                    crossarm_message = "Crossed Arms Detected"
                    (text_width, text_height), _ = cv2.getTextSize(crossarm_message, cv2.FONT_HERSHEY_SIMPLEX, text_size, text_thickness)
                    box_top_right = (frame_width - text_width - 20, 100)  # Positioning at the top right
                    box_bottom_left = (frame_width - 10, 100 + text_height + 10)
                    overlay = frame.copy()
                    cv2.rectangle(overlay, box_top_right, box_bottom_left, background_color, cv2.FILLED)
                    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                    text_position = (box_top_right[0] + 5, box_bottom_left[1] - 5)
                    cv2.putText(frame, crossarm_message, text_position, cv2.FONT_HERSHEY_SIMPLEX, text_size, text_color, text_thickness)


            else:
                if last_known_crossed_arms is not None:  # Pose ended
                    duration = current_time - crossed_arms_start_time
                    detections.append((crossed_arms_start_time, f"Crossed Arms duration: {duration:.1f}s"))
                    pose_durations.append({"pose": "Crossed Arms", "duration": duration})
                    last_known_crossed_arms = None  # Reset pose status

            # Hands Behind the Head Detection
            elbow_angle_condition = 20 <= left_elbow_angle <= 60 and 20 <= right_elbow_angle <= 60
            wrist_ear_dist_condition = 20 <= ear_wrist_dist_r[0] <= 40 and 20 <= ear_wrist_dist_l[0] <= 40

            if elbow_angle_condition and wrist_ear_dist_condition:
                if last_known_hands_behind_head is None:  # Pose just started
                    last_known_hands_behind_head = "Hands Behind the Head"
                    hands_behind_head_start_time = current_time
                # cv2.putText(frame, "Hands Behind the Head", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                if show_display:
                    handhead_message = "Hands Behind the Head"
                    (text_width, text_height), _ = cv2.getTextSize(handhead_message, cv2.FONT_HERSHEY_SIMPLEX, text_size, text_thickness)
                    box_top_right = (frame_width - text_width - 20, 100)  # Positioning at the top right
                    box_bottom_left = (frame_width - 10, 100 + text_height + 10)
                    overlay = frame.copy()
                    cv2.rectangle(overlay, box_top_right, box_bottom_left, background_color, cv2.FILLED)
                    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                    text_position = (box_top_right[0] + 5, box_bottom_left[1] - 5)
                    cv2.putText(frame, handhead_message, text_position, cv2.FONT_HERSHEY_SIMPLEX, text_size, text_color, text_thickness)

            else:
                if last_known_hands_behind_head is not None:  # Pose ended
                    duration = current_time - hands_behind_head_start_time
                    detections.append((hands_behind_head_start_time, f"Hands Behind the Head duration: {duration:.1f}s"))
                    pose_durations.append({"pose": "Hands Behind the Head", "duration": duration})
 
                    last_known_hands_behind_head = None  # Reset pose status

            






        angle_text_string = 'Neck : ' + str(int(neck_inclination)) + '  Torso : ' + str(int(torso_inclination))

        # Determine whether good posture or bad posture.
        # The threshold angles have been set based on intuition.
        if neck_inclination >= 24 and neck_inclination <=34  and torso_inclination >=120 and torso_inclination <=145:
            bad_frames = 0
            good_frames += 1


        else:
            good_frames = 0
            bad_frames += 1


        # Calculate the time of remaining in a particular posture.
        # good_time = (1 / fps) * good_frames
        # bad_time =  (1 / fps) * bad_frames

        # Pose time.
        
        current_time = time.time()

        if good_frames:
            if last_known_posture != "Good posture":
                is_bad_posture_detected = False
                # Calculate duration of the previous posture
                posture_duration = current_time - last_posture_change_time
                if last_known_posture:  # Check if there was a previous posture
                    detections.append((last_posture_change_time, f"{last_known_posture} duration: {posture_duration:.1f}s"))
                    posture_detection.append({"timestamp": last_posture_change_time, "message": f"{last_known_posture} duration: {posture_duration:.1f}s"})
                
                # Update for new posture
                last_known_posture = "Good posture"
                last_posture_change_time = current_time

            # Real-time display of current posture time
            good_time = current_time - last_posture_change_time
            text = f"Good Posture Time : {good_time:.1f}s"
            color = (0, 255, 0)  # Green for good posture       

        else:  # Assume posture is bad if not good
            if last_known_posture != "Bad posture":
                # Calculate duration of the previous posture
                posture_duration = current_time - last_posture_change_time
                if last_known_posture:  # This checks to ensure it's not the first frame of detection
                    detections.append((last_posture_change_time, f"{last_known_posture} duration: {posture_duration:.1f}s"))
                    posture_detection.append({"timestamp": last_posture_change_time, "message": f"{last_known_posture} duration: {posture_duration:.1f}s"})
                # Update for new posture
                last_known_posture = "Bad posture"
                last_posture_change_time = current_time

            # Real-time display of current posture time
            bad_time = current_time - last_posture_change_time
            if bad_time > 10:
                is_bad_posture_detected = True
                # print("You are in a bad posture for a long time")
            else :
                is_bad_posture_detected= False
            text = f"Bad Posture Time : {bad_time:.1f}s"
            color = (0, 0, 255)  # Red for bad posture
         
        if show_display:
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)

            # Define the box's coordinates based on the text size
            box_top_left = (10, h - 30 - text_height)
            box_bottom_right = (box_top_left[0] + text_width + 10, h - 10)

            # Draw a semi-transparent rectangle as the background
            overlay = frame.copy()
            cv2.rectangle(overlay, box_top_left, box_bottom_right, color, cv2.FILLED)
            alpha = 0.4  # Transparency factor
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # Adjust text position to be inside the box
            text_position = (box_top_left[0] + 5, box_bottom_right[1] - 5)

            # Draw the text on top of the rectangle
            cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)  # White text for readability


        # Write frames.
        video_output.write(frame)

        # # Display the processed frame.
        # cv2.imshow("Posture Detection", frame)

        # Process frame for face mesh and hand landmarks using MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        hand_results = hands.process(frame_rgb)
        face_results = face_mesh.process(frame_rgb)
        frame_rgb.flags.writeable = True

        # ---------------------------- Emotion Detection  --------------------------------------------
    
        

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face_img = gray[y:y + h, x:x + w]
            face_img = cv2.resize(face_img, (48, 48))
            img = extract_features(face_img)
            pred = model.predict(img, verbose=0)
            prediction_label = labels[pred.argmax()]

            current_time = time.time()

        # Check if the emotion has changed
            if prediction_label != last_known_emotion:
                # Calculate the duration of the last known emotion
                if last_known_emotion is not None:
                    emotion_duration = current_time - emotion_start_time[last_known_emotion]
                    duration_message = f"{last_known_emotion} emotion duration: {emotion_duration:.2f}s"
                    # print(duration_message)  # Or append it to detections with a timestamp
                    detections.append((emotion_start_time[last_known_emotion], duration_message))
                    emotion_detection.append({"timestamp": emotion_start_time[last_known_emotion], "message": duration_message})

                # Update the start time for the new emotion
                emotion_start_time[prediction_label] = current_time
                last_known_emotion = prediction_label
                # detections.append((current_time, prediction_label))

            # Yawning and Surprise Detection
    
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    lip_distance, upper_lip_point, lower_lip_point = get_lip_distance(face_landmarks.landmark, frame.shape[1], frame.shape[0])
                    
                    # Check for yawning
                    if lip_distance > yawning_threshold:
                        yawning_detected = 'the user is yawning'  # Use a new variable
                        detections.append((time.time(), yawning_detected))
                    
                    # Check for surprise
                    if lip_distance > surprise_threshold:
                        # Check if we're past the cooldown period from the last surprise
                        if last_surprise_time is None or (current_time - last_surprise_time) > surprise_cooldown:
                            surprise_counter += 1
                            detections.append((current_time, "the user is surprised"))
                            last_surprise_time = current_time

            # print(f"Total surprise detections: {surprise_counter}")

            # cv2.putText(frame, '% s' % (prediction_label), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            
            # Assuming 'frame' is your image/frame
            if show_display:
                emotion_display_text = f'{prediction_label}'
                # Choose a background color. Example: soft shade of blue (100, 100, 255), semi-transparent
                background_color = (100, 100, 255, 50)  # Last value is not applicable in OpenCV but indicates transparency concept
                text_color = (255, 255, 255)  # White text

                # Calculate text size to determine box size
                (text_width, text_height), baseline = cv2.getTextSize(emotion_display_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

                # Set coordinates for the rectangle background
                background_top_left = (5, 50)  # Adjust based on preference
                background_bottom_right = (background_top_left[0] + text_width + 10, background_top_left[1] + text_height + 10)

                # Draw semi-transparent rectangle
                overlay = frame.copy()
                cv2.rectangle(overlay, background_top_left, background_bottom_right, background_color, -1)

                # Applying the overlay with transparency
                alpha = 0.4  # Transparency factor
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                # Put text on top of the rectangle
                cv2.putText(frame, emotion_display_text, (background_top_left[0] + 5, background_bottom_right[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)


        # -------------  Blink Detection --------------------
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            # leftEyeHull = cv2.convexHull(leftEye) 
            # rightEyeHull = cv2.convexHull(rightEye)

            #draw eye landmark
            # cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            # cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            if ear < EYE_AR_THRESH:
                COUNTER += 1
            else:
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1
                COUNTER = 0
                    # detections.append((time.time(), "Blinks = " + str(TOTAL)))

            current_time = time.time()
            elapsed_time = current_time - start_time

             # Calculate blinks per minute every minute
            if elapsed_time >= 60:
                    # blinks_this_interval is TOTAL blinks detected in this 5-second interval
                    total_blinks_accumulated += TOTAL  # Add the number of blinks detected in this interval to the accumulated total
                    intervals_passed += 1  # Increment the number of intervals passed

                    # Calculate the average number of blinks per interval since the start
                    average_blinking = total_blinks_accumulated / intervals_passed

                    blink_data.append({
                        'time': intervals_passed,  # This could also be a timestamp
                        'average': average_blinking
                    })


                    print("Average blinks per interval:", average_blinking) 
                    detections.append((time.time(), "Average blinking time " + str(average_blinking)))

                    # Reset TOTAL for the next interval, not start_time or the accumulated totals
                    TOTAL = 0
                    start_time = current_time  # Reset start time for the next interval

            # cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
			# cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			# cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if show_display:
                    
                blink_display_text = "Blinks: {}".format(TOTAL)
                # Choose a background color. Example: light shade of green (100, 255, 100), semi-transparent
                background_color = (100, 255, 100, 50)  # The last value indicates the concept of transparency
                text_color = (255, 255, 255)  # White text for readability

                # Calculate text size to determine box size
                (text_width, text_height), baseline = cv2.getTextSize(blink_display_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        

                # Set coordinates for the rectangle background
                background_top_left = (5, 15)  # Adjust based on preference
                background_bottom_right = (background_top_left[0] + text_width + 10, background_top_left[1] + text_height + 10)

                # Draw semi-transparent rectangle
                overlay = frame.copy()
                cv2.rectangle(overlay, background_top_left, background_bottom_right, background_color, -1)

                # Applying the overlay with transparency
                alpha = 0.4  # Transparency factor
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                # Put text on top of the rectangle
                cv2.putText(frame, blink_display_text, (background_top_left[0] + 5, background_bottom_right[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)



        # ----------  Face Touch Detection  ----------------
        


    
        current_time = time.time()
        touching_face = False

        if hand_results.multi_hand_landmarks and face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    for landmark in hand_landmarks.landmark:
                        hx, hy = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                        for face_landmark in face_landmarks.landmark:
                            fx, fy = int(face_landmark.x * frame.shape[1]), int(face_landmark.y * frame.shape[0])
                            if abs(fx - hx) < 40 and abs(fy - hy) < 40:
                                touching_face = True
                                break
                        if touching_face:
                            break
                    if touching_face:
                        break

      
        if touching_face:
            if not is_touching_face:  # Face touch started now
                face_touch_start_time = current_time
                is_touching_face = True
                if face_touch_start_time > 10:
                    is_touching_face_long = True 
        else:
            if is_touching_face:  # Face touch ended
                is_touching_face_long = False
                face_touch_duration = current_time - face_touch_start_time
                detections.append((face_touch_start_time, f"Touching Face duration: {face_touch_duration:.2f}s"))
                pose_durations.append({"pose": "Face Touch", "duration": face_touch_duration})
                # print(f"Face touch ended. Duration: {face_touch_duration:.2f}s")  # Debug
                is_touching_face = False
                face_touch_start_time = None  # Reset for the next detection
            





        # --------  Hand Gesture Detection  --------------


        # model_path = 'D:\\Projects vs code\\Elevate\\Pose Estimation\\hand_gesture.pkl'

        # with open(model_path, 'rb') as f:
        #     hand_model = pickle.load(f)
        # # Load your trained model (adjust this path to where your model is stored)

        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Build the path to the pickle file
        model_path = os.path.join(script_dir, 'Pose Estimation', 'hand_gesturerc.pkl')

        # Load your trained model
        with open(model_path, 'rb') as f:
            hand_model = pickle.load(f)

        
        # Suppress specific UserWarning
        warnings.filterwarnings(action='ignore', category=UserWarning, message='.*X does not have valid feature names.*')

        # Process the frame
        frame_hand = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_hand.flags.writeable = False
        # frame_hand = cv2.flip(frame_hand, 1)
        results = hands.process(frame_hand)

        frame_hand.flags.writeable = True
        # frame_hand = cv2.cvtColor(frame_hand, cv2.COLOR_RGB2BGR)


        current_time = time.time()

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Draw landmarks
                

                # Prepare the hand landmarks as model input
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten().tolist()
                X = pd.DataFrame([landmarks])
                gesture_class = hand_model.predict(X)[0]
                
    
                if gesture_class != last_gesture_class or not is_gesture_detected:
                    # Gesture has changed
                    if is_gesture_detected:
                        # Calculate duration of the previous gesture
                        gesture_duration = current_time - gesture_start_time
                        detections.append((gesture_start_time, f"{last_gesture_class} duration: {gesture_duration:.2f}s"))
                        # print(f"Gesture {last_gesture_class} ended. Duration: {gesture_duration:.2f}s")

                    # Start tracking the new gesture
                    gesture_start_time = current_time
                    is_gesture_detected = True
                    last_gesture_class = gesture_class
                    # print(f"Gesture {gesture_class} started.")
                else:
                    # Update display with the current gesture without changing detection state
                    if show_display:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        wrist_coords = np.multiply(
                            np.array((hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y)), [640, 480]
                        ).astype(int)
                        cv2.putText(frame, gesture_class, (wrist_coords[0], wrist_coords[1]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            if is_gesture_detected:
                # End the current gesture if hands are no longer detected
                gesture_duration = current_time - gesture_start_time
                # print(f"Gesture {last_gesture_class} ended. Duration: {gesture_duration:.2f}s")
                detections.append((gesture_start_time, f"{last_gesture_class} duration: {gesture_duration:.2f}s"))
                is_gesture_detected = False
                gesture_start_time = None
                

        # Display the combined results
        # cv2.imshow('Multi-Feature Detector', frame)


        

        frame = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            is_running = False
            break

    webcam.release()  # Release the webcam
    video_output.release()  # Close the video writer
    cv2.destroyAllWindows()  # Close all OpenCV windows

    

    if last_known_posture:
        final_posture_duration = current_time - last_posture_change_time
        detections.append((last_posture_change_time, f"{last_known_posture} duration: {final_posture_duration:.1f}s"))
        posture_detection.append({"timestamp": last_posture_change_time, "message": f"{last_known_posture} duration: {final_posture_duration:.1f}s"})


    if last_known_emotion and last_known_emotion in emotion_start_time:
        # Calculate the duration for the last known emotion
        final_emotion_duration = current_time - emotion_start_time[last_known_emotion]
        combined_message = f"{last_known_emotion} duration: {final_emotion_duration:.2f}s"
        # Note: You're appending the start time of the emotion, consider if current_time is more appropriate for your use case
        detections.append((emotion_start_time[last_known_emotion], combined_message))
        emotion_detection.append({"timestamp": emotion_start_time[last_known_emotion], "message": combined_message})

    if not results.multi_hand_landmarks and is_gesture_detected:
        # This handles the case where the gesture ends because the hand left the scene
        gesture_duration = current_time - gesture_start_time
        detections.append((gesture_start_time, f"{last_gesture_class} duration: {gesture_duration:.2f}s"))
        is_gesture_detected = False
        gesture_start_time = None
        last_gesture_class = None
        print("Gesture detection reset due to no hand landmarks.")

    for detection_time, item in detections:
            dt_object = datetime.fromtimestamp(detection_time, tz=timezone.utc)
            # Format the datetime object to ISO 8601 format
            iso_format_time = dt_object.isoformat()
            # Print the formatted time and message
            combined_results.append({"timestamp": iso_format_time, "message":item})
            
            # print(f"{iso_format_time}: {item}")

    combined_results_sorted = sorted(combined_results, key=lambda x: datetime.fromisoformat(x['timestamp']))
    print(f"After sorting, combined_results_sorted: {combined_results_sorted}")

   


import openai
from Elevate import combined_results_sorted

@app.route('/progress')
@login_required
def progress():
    global combined_results_sorted
    global emotion_detection
    global posture_detection
    global pose_durations   
    global blink_data
    global speech_results_sorted
    time.sleep(10)
    # Your OpenAI API key
    openai.api_key = ''  # Replace with your OpenAI API Key (Prefereably paid version)

    
    emotion_durations = {}
    for entry in emotion_detection:
        # Split the message on " duration: " to separate the emotion from its duration
        parts = entry['message'].split(' duration: ')
        if len(parts) == 2:
            emotion, duration_str = parts
            # Remove 'emotion' from the emotion string if present
            emotion = emotion.replace(' emotion', '').strip()
            # Extract the duration value and convert it to float
            duration = float(duration_str.rstrip('s'))
            # Add the duration to the emotion's total in the dictionary
            if emotion in emotion_durations:
                emotion_durations[emotion] += duration
            else:
                emotion_durations[emotion] = duration

    emotion_summary = ""
    for detection in emotion_detection:
        timestamp = detection["timestamp"]
        message = detection["message"]
        emotion_summary += f"At {timestamp}, {message}\n"

# Ensure the summary is neatly formatted and user-friendly
    emotion_summary = emotion_summary.strip()

    emotion_prompt = f"""
    Analyze the emotional demeanor of the user during the virtual meeting based on the following observations:
    {emotion_summary}
    Provide feedback on how these emotions contribute to the user's virtual presence and suggest ways to manage emotions effectively.Keep it short.
    """

    try:
        emotion_analysis = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": emotion_prompt}]
        ).choices[0].message['content'].strip()
    except Exception as e:
        emotion_analysis = f"An error occurred while analyzing emotions: {str(e)}"
 

    posture_durations = {}
    for entry in posture_detection:
        parts = entry['message'].split(' duration: ')
        if len(parts) == 2:
            posture, duration_str = parts
            duration = float(duration_str.rstrip('s'))
            if posture in posture_durations:
                posture_durations[posture] += duration
            else:
                posture_durations[posture] = duration

    posture_summary = ""
    for detection in posture_detection:
        timestamp = detection["timestamp"]
        message = detection["message"]
        posture_summary += f"At {timestamp}, {message}\n"

# Ensure the summary is neatly formatted and user-friendly
    posture_summary = posture_summary.strip()

    posture_prompt = f"""
    Analyze the posture of the user during the virtual meeting based on the following observations:
    {posture_summary}
    If the posture observations are not empty, provide feedback on how these posture contribute to the user's virtual presence and suggest ways to manage posture effectively.Keep it short.
    """

    try:
        posture_analysis = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": posture_prompt}]
        ).choices[0].message['content'].strip()
    except Exception as e:
        posture_analysis = f"An error occurred while analyzing emotions: {str(e)}"
    
    from collections import defaultdict


    pose_duration_summary = "This analysis is based on the user's pose durations during the virtual meeting. Here are the details:\n"
    for pose in pose_durations:
        pose_duration_summary += f"- {pose['pose']}: {pose['duration']} seconds\n"

    pose_duration_summary += "\n If the pose details are not empty, please provide feedback on how these pose durations reflect on the user's engagement and body language during the virtual meeting.Keep it short."


    try:
        pose_analysis = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": pose_duration_summary}]
        ).choices[0].message['content'].strip()
    except Exception as e:
        pose_analysis = f"An error occurred while analyzing pose durations: {str(e)}"



    total_durations = defaultdict(float)
    for pose_info in pose_durations:
        total_durations[pose_info['pose']] += pose_info['duration']

    # Convert the results back to a list of dictionaries if needed
    aggregated_durations = [{'pose': pose, 'duration': duration} for pose, duration in total_durations.items()]

    

    
    # print(pose_durations)
    # print(posture_durations)
    # print(speech_results_sorted)

     # Assuming 'combined_results_sorted' contains all detected speech text
    full_text = " ".join([result['message'] for result in speech_results_sorted])
    # print("full text:",full_text)
    sentiment_result = get_sentiment_analysis(full_text)
    
    sentiment_summary = ""
    if sentiment_result:
        segments = sentiment_result["results"]["sentiments"]["segments"]
        for segment in segments:
            text = segment["text"]
            sentiment = segment["sentiment"]
            sentiment_score = segment["sentiment_score"]
            sentiment_summary += f'Text: "{text}"\nSentiment: {sentiment} (Score: {sentiment_score})\n\n'
    else:
        sentiment_summary = "Sentiment analysis failed."

    print(sentiment_summary)
    

    #SPEECH ANALYSIS
           
    speech_prompt = f"""   
    Analyze the speech sentiment and the filler words that the user is using during the virtual meeting based on the following observations: {sentiment_summary}
    
    If the observations are not empty provide feedback on how these sentiment and unwanted filler words if the user does them, contribute to the user's virtual presence and suggest ways to improve speech effectively. Mention with the
    analyzed speech where the user made errors if the user made any.Keep it short.
     """
                

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": speech_prompt}]
        )

        speech_analysis = response.choices[0].message['content'].strip()
    except Exception as e:
        analysis = f"An error occurred: {e}"
 

 #BLINK ANALYSIS
    if blink_data:  # Check if blink_data is not empty
        last_blink_average = blink_data[-1]['average']  # Access the last item's 'average' value
    else:
        last_blink_average = "No data"  # Handle case where there is no blink data

        
    blink_summary = "Blink Analysis Over Time:\n"
    for blink in blink_data:
        blink_summary += f"At interval {blink['time']}, the average blinks per minute was {blink['average']}.\n"

    blink_summary += "\nBased on this blinking data and if the data is not empty, please provide insights as to the user their average blinking time compared to a normal person during the virtual meeting.Keep it short."

    try:
        blink_analysis = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": blink_summary}]
        ).choices[0].message['content'].strip()
    except Exception as e:
        blink_analysis = f"An error occurred while analyzing blinking data: {str(e)}"



    existing_summary = "\n".join([f"At {result['timestamp']}, the observation was: {result['message']}" for result in combined_results_sorted])
    prompt = f"""   Give an analysis based on the detections provided tracking various aspects of user behavior during a virtual meeting, including posture, emotions, gestures, and eye blinking. The goal is to categorize these behaviors as positive or negative and offer feedback for improvement.

                    For the Posture Summarize the total duration of both good and bad postures detected. Emphasize the importance of maintaining a good posture for presenting a confident and engaged presence during virtual meetings. Suggest practical tips for improving posture if bad posture is prevalent.

                    For the Emotion Aggregate the durations of different emotions displayed and analyze the overall emotional demeanor. For a predominantly neutral emotion, explain that while neutrality is acceptable, displaying a cheerful demeanor can enhance engagement and warmth in interactions. If the dominant emotions are negative (e.g., sadness, anger), provide gentle feedback on the potential impact of these emotions on perceived confidence and suggest strategies for managing emotions effectively during meetings.

                    For the Gestures Identify any negative gestures observed, such as face touching, arm crossing, hands behind the head, or yawning. Discuss how these gestures may be perceived negatively in professional settings and offer advice on cultivating more positive body language. Highlight the durations of positive gestures (e.g., open hand, thumbs up, nodding, surprise) and commend the use of such gestures for their role in enhancing communication, showing affirmation, and indicating active listening.

                    For Eye Blink Comment on the average blinking rate and its variation throughout the meeting. If excessive blinking is noted, mention that it may indicate nervousness or discomfort and suggest ways to address this, perhaps by ensuring proper room lighting or taking breaks to rest the eyes.

                    For the Speech (Mention only If the speech is present) explain how the speech words in the detections aimed to realize what emotion, gesture, posture the user portrays when speaking certain sentences. the speech analysis analyses from the speech of the user what type of sentiment they are portraying. Give a transcript of the user speech with small small detections associated with it combining the sentiment analysis and the detections.

                    The analysis should be structured to provide a balanced overview of the user's performance, highlighting strengths while offering constructive feedback on areas for improvement. The aim is to encourage positive changes in behavior that will contribute to more effective and engaging virtual communication.    
                                    
                    Here are the detections : \n{existing_summary}\n\n  \n\nSentiment Analysis Summary (can be empty):\n{sentiment_summary}\n\n """
                
    print(prompt)



    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )

        analysis_content = response.choices[0].message['content'].strip()
        
    except Exception as e:
        analysis_content = f"An error occurred: {e}"

    analysis = analysis_content.replace("\n", "<br>").replace("1. ", "<strong>1. ").replace("2. ", "</strong><br><strong>2. ").replace("3. ", "</strong><br><strong>3. ").replace("4. ", "</strong><br><strong>4. ").replace("Overall,", "</strong><br><strong>Overall,")

    # Ensure the last <strong> tag is closed
    analysis += "</strong>"

    # Pass the formatted analysis content to the template
 
    return render_template('progress.html', last_blink_average=last_blink_average, speech_analysis = speech_analysis, posture_analysis=posture_analysis, emotion_analysis=emotion_analysis, blink_analysis=blink_analysis, pose_analysis = pose_analysis, combined_results_sorted=combined_results_sorted, analysis=analysis,  emotion_durations=emotion_durations, posture_durations=posture_durations, aggregated_durations=aggregated_durations, blink_data=blink_data)



@app.route('/logout', methods=['POST'])
def logout():
    # Your logout logic here
    
    logout_user()
    return redirect(url_for('login'))

import requests

DEEPGRAM_API_KEY = ""  # Replace this with your actual Deepgram API Key
DEEPGRAM_ENDPOINT = "https://api.deepgram.com/v1/read?sentiment=true&language=en"

def get_sentiment_analysis(text):
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "text": text
    }

    response = requests.post(DEEPGRAM_ENDPOINT, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}, Response: {response.text}")
        return None


@app.route('/process_speech_results', methods=['POST'])
def process_speech_results():
    global combined_results
    global speech_results_sorted
    data = request.get_json()
    for item in data:
        # Ensure the timestamp is in ISO 8601 format
        combined_results.append({"timestamp": item['timestamp'], "message": item['transcript']})
        speech_results_sorted.append({"timestamp": item['timestamp'], "message": item['transcript']})
       
        # print(f"{item['timestamp']}: {item['transcript']}")

    
    return jsonify({"message": "Data received successfully"})





@app.route('/check_bad_posture')
def check_bad_posture():
    global is_bad_posture_detected
    return jsonify({"isBadPosture": is_bad_posture_detected})

@app.route('/check_face_touching')
def check_face_touching():
    # Example logic to determine if the user has been touching their face too long
    # You need to implement how you track the duration of face touching
    global is_touching_face_long # Determine based on your logic

    return jsonify({"isTouchingFace": is_touching_face_long})


@app.route('/Elevate')
# @login_required
def Elevate():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')




from forms import LoginForm, SignupForm
from flask import redirect, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and user.check_password(form.password.data):
            login_user(user)
            return redirect(url_for('home'))  # Redirect to the main page
        flash('Invalid email or password')
    return render_template('login.html', form=form)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = SignupForm()
    if form.validate_on_submit():
        existing_user = User.query.filter_by(email=form.email.data).first()
        if existing_user is None:
            user = User(full_name=form.full_name.data, username=form.username.data,
                        email=form.email.data)
            user.set_password(form.password.data)
            db.session.add(user)
            db.session.commit()
            # Log the user in (you'll need to handle sessions)
            return redirect(url_for('home'))  # Redirect to the main page
        flash('A user already exists with that email.')
    return render_template('signup.html', form=form)

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    global is_running
    print("Before stopping, is_running =", is_running)
    is_running = False
    print("After stopping, is_running =", is_running)
    return jsonify({'success': True, 'message': 'Detection stopped successfully'})


@app.route('/')
def homepage():
    return redirect(url_for('login')) if not current_user.is_authenticated else redirect(url_for('home'))


@app.route('/home')

def home():
    return render_template('homepage.html')

@app.route('/index')
def detection():
    if current_user.is_authenticated:   
        return render_template('index.html')
    return redirect(url_for('login'))

@app.route('/deepgram')
@login_required
def deepgram():
    return render_template('deepgram.html')

@app.route('/features')
def feautures():
    return render_template('features.html')

@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')

@app.route('/toggle_display', methods=['POST'])
def toggle_display():
    global show_display
    show_display = not show_display
    return jsonify({'status': 'success', 'message': 'Display toggled successfully'})



if __name__=="__main__":
    app.run(debug=True, port=8081)



