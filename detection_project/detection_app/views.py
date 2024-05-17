from django.shortcuts import render
import cv2
import numpy as np
import time
import mediapipe as mp
from scipy.spatial import distance as dist
from imutils import face_utils
import dlib
from keras.models import model_from_json
import math as m


# Create your views here.

from django.shortcuts import render

def detection_system(request):
        # Your detection code goes here
        # You can render an HTML template to display the results




    # Initilize medipipe selfie segmentation class.
    mp_pose = mp.solutions.pose
    mp_holistic = mp.solutions.holistic

    def findDistance(x1, y1, x2, y2):
        dist = m.sqrt((x2-x1)**2+(y2-y1)**2)
        return dist

    # Calculate angle.
    def findAngle(x1, y1, x2, y2):
        theta = m.acos((y2 -y1)*(-y1) / (m.sqrt((x2 - x1)**2 + (y2 - y1)**2) * y1))
        degree = int(180/m.pi)*theta
        return degree

    def sendWarning():
        print('YOU ARE IN WRONG POSITION FOR LONG TIME')

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
    if not webcam.isOpened():
        print("Error: Camera is not accessible")
        exit()

    fps = int(webcam.get(cv2.CAP_PROP_FPS))
    width = int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Initialize video writer.
    video_output = cv2.VideoWriter('/content/output.mp4', fourcc, fps, frame_size)



    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()




    # Initialize variables for yawning detection
    yawning_start_time = None
    yawning_threshold = 27  # Lip distance threshold for yawning
    yawning_duration = 3    # Duration in seconds to confirm yawning
    surprise_threshold = 14

    # Load the emotion detection model
    json_file = open("D:/Projects vs code/Elevate/Face_Detection/emotiondetector.json", "r")
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights("D:/Projects vs code/Elevate/Face_Detection/emotiondetector.h5")

    # Initialize Haar Cascade, MediaPipe, and dlib
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    mp_hands = mp.solutions.hands
    mp_face_mesh = mp.solutions.face_mesh
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("Face_Detection/shape_predictor_68_face_landmarks.dat")

    # Blink detection setup
    EYE_AR_THRESH = 0.3
    EYE_AR_CONSEC_FRAMES = 3
    COUNTER = 0
    TOTAL = 0
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    def eye_aspect_ratio(eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

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


    # Webcam initialization and emotion labels

    labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

    def run_detection():
        global good_frames, bad_frames, COUNTER, TOTAL, EYE_AR_THRESH, EYE_AR_CONSEC_FRAMES
        while True:
            ret, frame = webcam.read()
            if not ret:
                break
            
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
                # Right shoulder
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

                
                
                    # Calculate distance between left shoulder and right shoulder points.
            offset = findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)

            # Assist to align the camera to point at the view of the person.
            # Offset threshold 30 is based on results obtained from analysis over 100 samples.

            # if offset < 300:
            #     cv2.putText(frame, str(int(offset)) + ' Aligned', (w - 150, 30), font, 0.9, green, 2)
            # else:
            #     cv2.putText(frame, str(int(offset)) + ' Not Aligned', (w - 150, 30), font, 0.9, red, 2)

            # Calculate angles.
            neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
            torso_inclination = findAngle(l_nose_x, l_nose_y, l_shldr_x, l_shldr_y)

            # Draw landmarks.
            # cv2.circle(frame, (l_shldr_x, l_shldr_y), 7, yellow, -1)
            # cv2.circle(frame, (l_ear_x, l_ear_y), 7, yellow, -1)

            # cv2.circle(frame, (r_shldr_x, r_shldr_y), 7, yellow, -1)
            # cv2.circle(frame, (r_ear_x, r_ear_y), 7, yellow, -1)

            # Let's take y - coordinate of P3 100px above x1,  for display elegance.
            # Although we are taking y = 0 while calculating angle between P1,P2,P3.
            # cv2.circle(frame, (l_shldr_x, l_shldr_y - 100), 7, yellow, -1)
            # cv2.circle(frame, (r_shldr_x, r_shldr_y - 100), 7, yellow, -1)
            # cv2.circle(frame, (r_shldr_x, r_shldr_y), 7, yellow, -1)
            # cv2.circle(frame, (l_nose_x, l_nose_y), 7, yellow, -1)

            # Similarly, here we are taking y - coordinate 100px above x1. Note that
            # you can take any value for y, not necessarily 100 or 200 pixels.
            # cv2.circle(frame, (l_nose_x, l_nose_y - 100), 7, blue, -1)

            # Put text, Posture and angle inclination.
            # Text string for display.
            angle_text_string = 'Neck : ' + str(int(neck_inclination)) + '  Torso : ' + str(int(torso_inclination))

            # Determine whether good posture or bad posture.
            # The threshold angles have been set based on intuition.
            if neck_inclination >= 24 and neck_inclination <=34  and torso_inclination >=120 and torso_inclination <=145:
                bad_frames = 0
                good_frames += 1
        # writing in a good frame posture 
                # cv2.putText(frame, angle_text_string, (10, 30), font, 0.9, light_green, 2)
                # cv2.putText(frame, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, light_green, 2)
                # cv2.putText(frame, str(int(torso_inclination)), (l_nose_x + 10, l_nose_y), font, 0.9, light_green, 2)

                # # Join landmarks.
                # cv2.line(frame, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), green, 4)
                # cv2.line(frame, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), green, 4)
                # cv2.line(frame, (l_nose_x, l_nose_y), (l_shldr_x, l_shldr_y), green, 4)
                # cv2.line(frame, (l_nose_x, l_nose_y), (l_nose_x, l_nose_y - 100), green, 4)

                # cv2.line(frame, (r_shldr_x, r_shldr_y), (r_ear_x, r_ear_y), green, 4)
                # cv2.line(frame, (r_shldr_x, r_shldr_y), (r_shldr_x, r_shldr_y - 100), green, 4)
                # cv2.line(frame, (l_nose_x, l_nose_y), (r_shldr_x, r_shldr_y), green, 4)
                # cv2.line(frame, (l_nose_x, l_nose_y), (l_nose_x, l_nose_y - 100), green, 4)

            else:
                good_frames = 0
                bad_frames += 1

                # cv2.putText(frame, angle_text_string, (10, 30), font, 0.9, red, 2)
                # cv2.putText(frame, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, red, 2)
                # cv2.putText(frame, str(int(neck_inclination)), (r_shldr_x + 10, r_shldr_y), font, 0.9, red, 2)
                # cv2.putText(frame, str(int(torso_inclination)), (l_nose_x + 10, l_nose_y), font, 0.9, red, 2)

                # # Join landmarks.
                # cv2.line(frame, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), red, 4)
                # cv2.line(frame, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), red, 4)
                # cv2.line(frame, (l_nose_x, l_nose_y), (l_shldr_x, l_shldr_y), red, 4)
                # cv2.line(frame, (l_nose_x, l_nose_y), (l_nose_x, l_nose_y - 100), red, 4)

                # cv2.line(frame, (r_shldr_x, r_shldr_y), (r_ear_x, r_ear_y), red, 4)
                # cv2.line(frame, (r_shldr_x, r_shldr_y), (r_shldr_x, r_shldr_y - 100), red, 4)
                # cv2.line(frame, (l_nose_x, l_nose_y), (r_shldr_x, r_shldr_y), red, 4)
                # cv2.line(frame, (l_nose_x, l_nose_y), (l_nose_x, l_nose_y - 100), red, 4)

            # Calculate the time of remaining in a particular posture.
            good_time = (1 / fps) * good_frames
            bad_time =  (1 / fps) * bad_frames

            # Pose time.
            if good_time > 0:
                time_string_good = 'Good Posture Time : ' + str(round(good_time, 1)) + 's'
                cv2.putText(frame, time_string_good, (10, h - 20), font, 0.9, green, 2)
            else:
                time_string_bad = 'Bad Posture Time : ' + str(round(bad_time, 1)) + 's'
                cv2.putText(frame, time_string_bad, (10, h - 20), font, 0.9, red, 2)

            # If you stay in bad posture for more than 3 minutes (180s) send an alert.
            if bad_time > 30:
                sendWarning()
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

            # Emotion Detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                face_img = gray[y:y + h, x:x + w]
                face_img = cv2.resize(face_img, (48, 48))
                img = extract_features(face_img)
                pred = model.predict(img, verbose=0)
                prediction_label = labels[pred.argmax()]

            # Yawning and Surprise Detection
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    # draw_landmarks(frame, face_landmarks.landmark)
                    lip_distance, upper_lip_point, lower_lip_point = get_lip_distance(face_landmarks.landmark, frame.shape[1], frame.shape[0])
                    #draw line for lip distance
                    # cv2.line(frame, upper_lip_point, lower_lip_point, (255, 0, 0), 2)
                    cv2.putText(frame, f'Lip Distance: {lip_distance:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    if lip_distance > yawning_threshold:
                        if yawning_start_time is None:
                            yawning_start_time = time.time()
                        elif (time.time() - yawning_start_time) >= yawning_duration:
                            prediction_label = 'yawning'
                    else:
                        yawning_start_time = None
                        if lip_distance > surprise_threshold:
                            prediction_label = 'surprise'

            cv2.putText(frame, '% s' % (prediction_label), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Blink Detection
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

                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                #draw eye landmark
                # cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                # cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                if ear < EYE_AR_THRESH:
                    COUNTER += 1
                else:
                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        TOTAL += 1
                        COUNTER = 0

                cv2.putText(frame, "Blinks: {}".format(TOTAL), (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # cv2.putText(frame, "EAR: {:.2f}".format(ear), (400, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Face Touch Detection
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
                        cv2.putText(frame, 'Touching Face!', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        break

            # Display the combined results
            cv2.imshow('Multi-Feature Detector', frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        # Release resources
        webcam.release()
        cv2.destroyAllWindows()

    
    return render(request, 'detection_app/detection.html')
