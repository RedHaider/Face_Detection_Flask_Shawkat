import cv2
import face_recognition
import numpy as np
import requests
import datetime
import pickle
import time

# Load known face encodings and names
with open('trained_faces.pkl', 'rb') as f:
    known_face_encodings, known_face_names = pickle.load(f)

def recognize_faces(frame, known_face_encodings, known_face_names):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame, model='hog')
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    recognized_names = []

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        recognized_names.append(name)

        # Draw rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw label with name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    return frame, recognized_names

def mark_attendance(names):
    attendance_log = []
    now = datetime.datetime.now()
    for name in names:
        attendance_log.append({'name': name, 'time': now.strftime("%Y-%m-%d %H:%M:%S")})
        print(f"Attendance marked for {name} at {now}")
    return attendance_log

def upload_attendance(attendance_log):
    server_url = "http://localhost/attendance/upload_attendance.php"
    data = {'attendance_log': str(attendance_log)}

    response = requests.post(server_url, data=data)

    if response.status_code == 200:
        print("Attendance data uploaded successfully")
    else:
        print("Failed to upload attendance data")

def main():
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            continue

        frame, recognized_names = recognize_faces(frame, known_face_encodings, known_face_names)

        if recognized_names:
            print(f'Recognized faces: {recognized_names}')
            attendance_log = mark_attendance(recognized_names)
            upload_attendance(attendance_log)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()