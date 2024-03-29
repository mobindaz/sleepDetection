import cv2
import mediapipe as mp
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("model.h5")

# Initialize MediaPipe Face Detection and Pose models
mp_face_detection = mp.solutions.face_detection
mp_pose = mp.solutions.pose

face_detection = mp_face_detection.FaceDetection()
pose = mp_pose.Pose()

# Initialize a dictionary to store names corresponding to each detected face
names = {}


# Function to detect if the person is sleeping
def is_sleeping(landmarks):
    # You can define your own criteria for sleeping posture
    # For example, if the y-coordinate of the head landmark is lower than a certain threshold
    head_y = landmarks[mp_pose.PoseLandmark.NOSE.value].y
    print(head_y)
    if head_y > 0.55:  # Adjust the threshold as needed
        return True
    else:
        return False


# Capture video from webcam
cap = cv2.VideoCapture(0)

person_counter = 0  # Counter to assign unique names to each detected person

# Initialize LabelEncoder
label_encoder = LabelEncoder()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image with MediaPipe Face Detection
    results_face = face_detection.process(rgb_frame)
    if results_face.detections:
        for detection in results_face.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            # Draw rectangle around the face
            cv2.rectangle(frame, bbox, (0, 255, 0), 2)

            # Process the image with MediaPipe Pose
            results_pose = pose.process(rgb_frame)

            if results_pose.pose_landmarks:
                landmarks = results_pose.pose_landmarks.landmark

                # Check if the person is sleeping
                if is_sleeping(landmarks):
                    # Preprocess the image
                    face_img = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
                    face_img = cv2.resize(face_img, (224, 224))
                    face_img = np.expand_dims(face_img, axis=0)
                    face_img = face_img / 255.0

                    # Predict the name using the loaded model
                    predicted_label_index = np.argmax(model.predict(face_img))
                    predicted_name = label_encoder.inverse_transform([predicted_label_index])[0]

                    # Print the name and sleeping status on the screen
                    cv2.putText(frame, f"{predicted_name} is sleeping", (bbox[0], bbox[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
