import cv2
import dlib
import pyttsx3
import os
from datetime import datetime
import pandas as pd
import time
import numpy as np

# Initialize the Text-to-Speech engine
engine = pyttsx3.init()

# Load the face detector and face recognizer
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Load the face landmark predictor
face_recognition_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")  # Load the face recognition model

# Initialize known faces and names
known_faces = []
known_names = []

# Track attendance to avoid marking the same person multiple times
attendance_marked = set()

# Load known faces and their names
def load_known_faces():
    global known_faces, known_names

    image_folder = "known_faces/"
    if not os.path.exists(image_folder):
        print("No known faces folder found. You can still test with unknown faces.")
        return

    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(image_folder, filename)
            try:
                image = cv2.imread(image_path)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = detector(gray)

                for face in faces:
                    landmarks = predictor(image, face)
                    face_encoding = face_recognition_model.compute_face_descriptor(image, landmarks)
                    known_faces.append(np.array(face_encoding))
                    known_names.append(os.path.splitext(filename)[0])  # name is the image filename without extension
            except Exception as e:
                print(f"Error processing image {filename}: {e}")

# Function to mark attendance
def mark_attendance(name):
    if name in attendance_marked:
        return  # Skip if attendance already marked

    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"Attendance marked for {name} at {current_time}")

    # Create or append to the attendance CSV file
    if not os.path.exists('attendance.csv'):
        df = pd.DataFrame(columns=['Name', 'Timestamp'])
        df.to_csv('attendance.csv', index=False)  # Create the file if it doesn't exist

    df = pd.read_csv('attendance.csv')
    df = pd.concat([df, pd.DataFrame({'Name': [name], 'Timestamp': [current_time]})], ignore_index=True)

    df.to_csv('attendance.csv', index=False)

    # Voice feedback
    engine.say(f"Attendance recorded for {name}")
    engine.runAndWait()
    engine.say("You may now proceed.")  # Optional message
    engine.runAndWait()

    # Mark this person as attended
    attendance_marked.add(name)

# Function to calculate Euclidean distance between two face encodings
def euclidean_distance(face1, face2):
    return np.linalg.norm(face1 - face2)

# Main face recognition process with laptop camera
def start_face_recognition():
    load_known_faces()

    video_capture = cv2.VideoCapture(0)  # 0 is the default camera
    
    # Check if the camera is working
    if not video_capture.isOpened():
        print("Error: Unable to access the camera.")
        return

    attendees_limit = 110  # Set limit for number of attendees you want to mark

    while True:
        ret, frame = video_capture.read()
        
        # Ensure a valid frame is returned
        if not ret:
            print("Failed to grab frame from camera.")
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find faces in the current frame
        faces = detector(gray)

        if len(faces) == 0:
            print("No faces detected in the frame.")
        
        for face in faces:
            # Get landmarks and face encoding
            try:
                landmarks = predictor(frame, face)
                face_encoding = face_recognition_model.compute_face_descriptor(frame, landmarks)
                face_encoding = np.array(face_encoding)

                # Compare this face with known faces
                min_distance = float('inf')
                name = "Unknown"
                
                for i, known_face in enumerate(known_faces):
                    distance = euclidean_distance(face_encoding, known_face)
                    if distance < min_distance:  # If this face matches better
                        min_distance = distance
                        name = known_names[i]

                # If the distance is below a threshold, consider it a match
                if min_distance < 0.6:  # Adjust this threshold if necessary
                    mark_attendance(name)

                # Draw rectangle around the face
                cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 2)

                # Display the name of the recognized person
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (face.left() + 6, face.bottom() - 6), font, 0.5, (255, 255, 255), 1)

            except Exception as e:
                print(f"Error in face recognition: {e}")
                continue

        # Display the resulting image (real-time video feed)
        cv2.imshow('Video', frame)

        # Press 'q' to quit the webcam feed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Exit after marking the desired number of attendees
        if len(attendance_marked) >= attendees_limit:
            print("Maximum number of attendees reached.")
            break

        # Optional: small delay to prevent continuous processing
        time.sleep(4)

    # Release the camera and close all OpenCV windows
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Create attendance file if it doesn't exist
    if not os.path.exists('attendance.csv'):
        pd.DataFrame(columns=['Name', 'Timestamp']).to_csv('attendance.csv', index=False)

    # Start face recognition with the laptop camera
    start_face_recognition()
