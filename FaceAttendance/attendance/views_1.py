# Capture Live Video for Attendance
import cv2
import face_recognition
import numpy as np
import dlib
import pyttsx3
import threading
import queue
from imutils import face_utils
from django.http import StreamingHttpResponse
from django.utils import timezone
from django.shortcuts import render
from .models import Employee, Attendance
import os
from django.http import JsonResponse



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTOR_PATH = os.path.join(BASE_DIR, "shape_predictor_68_face_landmarks.dat")

if not os.path.exists(PREDICTOR_PATH):
    raise FileNotFoundError(f"File not found: {PREDICTOR_PATH}")

predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()

# Create a queue to handle speech requests
speech_queue = queue.Queue()

def speech_worker():
    """Speech synthesis worker that runs in the background to process queued speech requests."""
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    
    while True:
        text = speech_queue.get()  # Wait for a speech request
        if text is None:  
            break  # Stop the worker thread if None is received
        
        engine.say(text)
        engine.runAndWait()
        speech_queue.task_done()

# Start the speech worker thread
speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()

def speak_text(text):
    """Add a speech request to the queue."""
    speech_queue.put(text)

def eye_aspect_ratio(eye):
    """Calculate the eye aspect ratio to detect blinks"""
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

def detect_blink(gray, face):
    """Detect if a person blinks"""
    shape = predictor(gray, face)
    shape = face_utils.shape_to_np(shape)
    left_eye = shape[42:48]
    right_eye = shape[36:42]
    ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
    return ear < 0.2

def generate_frames():
    video_capture = cv2.VideoCapture(0)

    employees = Employee.objects.all()
    known_face_encodings = []
    known_face_names = []
    known_employee_ids = [] 
    known_employee_paths = [] 

    for employee in employees:
        if employee.face_encoding:
            encoding_array = np.frombuffer(employee.face_encoding, dtype=np.float64)
            if encoding_array.size == 128:
                known_face_encodings.append(encoding_array)
                known_face_names.append(employee.name)
                known_employee_ids.append(employee.id)  # Store the ID
                known_employee_paths.append(employee.image)  # Store the image path

    recognized_faces = set()

    while True:
        success, frame = video_capture.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            detect_blink(gray, face)

        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            box_color = (0, 0, 255)

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                employee_id = known_employee_ids[first_match_index]  # Get the correct ID
                employee_image = known_employee_paths[first_match_index]  # Get the correct ID

                box_color = (0, 255, 0)

                if name not in recognized_faces:
                    recognized_faces.add(name)
                    speak_text(f"Welcome {name}")

                    try:
                        print(f"Employee Image: {employee_id}")

                        attendance = Attendance.objects.create(emp_id=employee_id, time_in=timezone.now())
                        print(f"Attendance marked at {attendance.time_in}")

                    except Exception as e:
                        print(f"Error marking attendance for {name}: {e}")

            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    video_capture.release()

def video_feed(request):
    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def live_feed_page(request):
    return render(request, 'attendance/live_feed.html')

def generate_and_store_face_encoding(request):
    employees = Employee.objects.all()

    for employee in employees:
        try:
            # Assuming 'image' is the file field of Employee model
            image_path = employee.image.path
            # Load the image from the file path
            image = face_recognition.load_image_file(image_path)

            # Generate face encodings for the image (assuming one face per image)
            face_encodings = face_recognition.face_encodings(image)

            if face_encodings:
                # Convert the face encoding into bytes to store in the database
                face_encoding_bytes = face_encodings[0].tobytes()

                # Update employee with the generated face encoding
                employee.face_encoding = face_encoding_bytes
                employee.save()

                print(f"Face encoding saved for {employee.name}")
            else:
                print(f"No face found for {employee.name}")

        except Exception as e:
            print(f"Error processing image for {employee.name}: {e}")

    return JsonResponse({"message": "Face encodings generated and stored."})


def attendance_view(request):
    return render(request, 'attendance/attendance.html')

def recognize_face(frame):
    known_faces = []
    employees = Employee.objects.all()
    
    for emp in employees:
        known_faces.append((emp.emp_id, np.frombuffer(emp.face_encoding, dtype=np.float64)))

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces([f[1] for f in known_faces], face_encoding)
        face_distances = face_recognition.face_distance([f[1] for f in known_faces], face_encoding)
        
        best_match_index = np.argmin(face_distances) if face_distances.size else -1
        
        if best_match_index >= 0 and matches[best_match_index]:
            emp_id = known_faces[best_match_index][0]
            return emp_id, location
    
    return None, None

def live_camera_feed(request):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        emp_id, location = recognize_face(frame)
        if emp_id:
            return JsonResponse({"status": "success", "emp_id": emp_id})
    
    cap.release()
    return JsonResponse({"status": "no_match"})
