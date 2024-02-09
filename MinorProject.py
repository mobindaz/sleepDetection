import cv2
import mediapipe as mp

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
    if head_y < 0.3:  # Adjust the threshold as needed
        return True
    else:
        return False

# Capture video from webcam
cap = cv2.VideoCapture(0)

person_counter = 0  # Counter to assign unique names to each detected person

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

            # Get the unique name for the person
            person_name = names.get(person_counter, f"Person{person_counter+1}")
            person_counter += 1

            # Process the image with MediaPipe Pose
            results_pose = pose.process(rgb_frame)

            if results_pose.pose_landmarks:
                landmarks = results_pose.pose_landmarks.landmark

                # Check if the person is sleeping
                if is_sleeping(landmarks):
                    # Print the name and sleeping status on the screen
                    cv2.putText(frame, f"{person_name} is sleeping", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    # Save the current frame as an image
                    cv2.imwrite(f'{person_name}_sleeping.jpg', frame)

            # Save the name for future reference
            names[person_counter - 1] = person_name

    # Display the frame
    cv2.imshow('Frame', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
qq

# import cv2
# import mediapipe as mp
#
# # Initialize MediaPipe Face Detection and Pose models
# mp_face_detection = mp.solutions.face_detection
# mp_pose = mp.solutions.pose
#
# face_detection = mp_face_detection.FaceDetection()
# pose = mp_pose.Pose()
#
# # Function to detect if the person is sleeping
# def is_sleeping(landmarks):
#     # You can define your own criteria for sleeping posture
#     # For example, if the y-coordinate of the head landmark is lower than a certain threshold
#     head_y = landmarks[mp_pose.PoseLandmark.NOSE.value].y
#     if head_y < 0.3:  # Adjust the threshold as needed
#         return True
#     else:
#         return False
#
# # Capture video from webcam
# cap = cv2.VideoCapture(0)
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # Convert the image to RGB
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#     # Process the image with MediaPipe Face Detection
#     results_face = face_detection.process(rgb_frame)
#     if results_face.detections:
#         for detection in results_face.detections:
#             bboxC = detection.location_data.relative_bounding_box
#             ih, iw, _ = frame.shape
#             bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
#
#             # Draw rectangle around the face
#             cv2.rectangle(frame, bbox, (0, 255, 0), 2)
#
#             # Process the image with MediaPipe Pose
#             results_pose = pose.process(rgb_frame)
#
#             if results_pose.pose_landmarks:
#                 landmarks = results_pose.pose_landmarks.landmark
#
#                 # Check if the person is sleeping
#                 if is_sleeping(landmarks):
#                     # Save the current frame as a photo
#                     cv2.imwrite('sleeping_person.jpg', frame)
#
#     # Display the frame
#     cv2.imshow('Frame', frame)
#
#     # Exit if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Release resources
# cap.release()
# cv2.destroyAllWindows()
