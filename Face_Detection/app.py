from flask import Flask, render_template, Response
import cv2
import face_recognition
import pickle

app = Flask(__name__)

# Load the trained faces
def load_trained_faces(filename='trained_faces.pkl'):
    with open(filename, 'rb') as f:
        known_face_encodings, known_face_names = pickle.load(f)
    return known_face_encodings, known_face_names

known_face_encodings, known_face_names = load_trained_faces()

# Initialize video capture
video_capture = cv2.VideoCapture(1)  # Use 0 for default camera

def generate_frames():
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture image")
            continue

        # Check the frame format
        if frame is None:
            print("Captured frame is None")
            continue

        # Print frame details for debugging
        # print(f"Frame shape: {frame.shape}")
        # print(f"Frame dtype: {frame.dtype}")

        # Ensure the frame is in 8-bit format
        if frame.dtype != 'uint8':
            print("Unsupported image type:", frame.dtype)
            continue

        # Convert the frame to RGB
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error converting frame to RGB: {e}")
            continue

        # Perform face detection and recognition
        try:
            face_locations = face_recognition.face_locations(rgb_frame, model='hog')
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        except Exception as e:
            print(f"Error in face recognition: {e}")
            continue

        # Draw rectangles and labels
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                # print(name)

            # Draw rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Draw label with name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("Failed to encode image")
            continue

        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@

if __name__ == "__main__":
    app.run(debug=Tr