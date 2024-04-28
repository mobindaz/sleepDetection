import cv2
from scipy.spatial import distance as dist

# Load the pre-trained face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the pre-trained eye cascade classifier
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to check if the person is sleeping
def is_sleeping(ear, threshold=0.25):  
    return ear < threshold

# Function to save image
def save_image(frame, filename):
    cv2.imwrite(filename, frame)

# Open the webcam
cap = cv2.VideoCapture(0)  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract eye regions
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes in the face region
        eyes = eye_cascade.detectMultiScale(roi_gray)

        # If there are no eyes detected, continue to the next face
        if len(eyes) < 2:
            continue

        # Extract the left and right eye regions
        left_eye = []
        right_eye = []
        for (ex, ey, ew, eh) in eyes:
            eye_center = (x + ex + ew // 2, y + ey + eh // 2)
            if eye_center[0] < x + w // 2:
                left_eye.append((ex, ey, ew, eh))
            else:
                right_eye.append((ex, ey, ew, eh))

        # Check if there are enough eye detections to calculate EAR
        if len(left_eye) >= 6 and len(right_eye) >= 6:
            # Calculate Eye Aspect Ratio (EAR) for each eye
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)

            # Calculate the average EAR
            ear = (left_ear + right_ear) / 2.0

            # Check if the face is looking down (e.g., eyes are near the bottom of the face)
            if (y + h) - (2 * min(left_eye[0][3], right_eye[0][3])) > y + h // 2:
                # Display text indicating sleeping
                cv2.putText(frame, "Sleeping", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # Save image
                save_image(frame, 'sleeping_image.jpg')

    # Display the frame
    cv2.imshow("Classroom Monitoring", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
