from flask import Flask, render_template, Response, jsonify
import cv2
import face_recognition
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained face encodings and names
def load_trained_faces(filename='trained_faces.pkl'):
    with open(filename, 'rb') as f:
        known_face_encodings, known_face_names = pickle.load(f)
    return known_face_encodings, known_face_names

known_face_encodings, known_face_names = load_trained_faces()

# Initialize video capture
video_capture = cv2.VideoCapture(0)

def recognize_faces(frame):
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

    return frame

def gen_frames():
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        frame = recognize_faces(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
