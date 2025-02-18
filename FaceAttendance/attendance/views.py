# Capture Live Video for Attendance
# import cv2
# import face_recognition
# from django.http import JsonResponse
# from django.shortcuts import render
# import numpy as np
# from django.contrib.auth.models import User
# from .models import Employee, Attendance
# from django.utils import timezone  # Import timezone for time management
# import dlib
# from imutils import face_utils
# import os

import cv2
import face_recognition
import numpy as np
import dlib
from imutils import face_utils
from django.http import StreamingHttpResponse
from django.utils import timezone
from django.shortcuts import render
from .models import Employee, Attendance
import os

# Load the face detector
# detector = dlib.get_frontal_face_detector()
# predictor_path = "shape_predictor_68_face_landmarks.dat"  # Make sure the file is in your project
# predictor = dlib.shape_predictor(predictor_path)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the directory of views.py
PREDICTOR_PATH = os.path.join(BASE_DIR, "shape_predictor_68_face_landmarks.dat")

if not os.path.exists(PREDICTOR_PATH):
    raise FileNotFoundError(f"File not found: {PREDICTOR_PATH}")

predictor = dlib.shape_predictor(PREDICTOR_PATH)
# Load face detector and eye landmark predictor
detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download required


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

    left_EAR = eye_aspect_ratio(left_eye)
    right_EAR = eye_aspect_ratio(right_eye)

    ear = (left_EAR + right_EAR) / 2.0
    return ear < 0.2  # If eyes closed, return True

def generate_frames():
    video_capture = cv2.VideoCapture(0)

    employees = Employee.objects.all()
    known_face_encodings = []
    known_face_names = []

    for employee in employees:
        if employee.face_encoding:
            encoding_array = np.frombuffer(employee.face_encoding, dtype=np.float64)
            if encoding_array.size == 128:
                known_face_encodings.append(encoding_array)
                known_face_names.append(employee.name)

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

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

                try:
                    attendance = Attendance.objects.create(user_id='001', time_in=timezone.now())
                    print(f"Attendance marked at {attendance.time_in}")

                except Exception as e:
                    print(f"Error marking attendance for {name}: {e}")

            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
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


# previus code
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the directory of views.py
# PREDICTOR_PATH = os.path.join(BASE_DIR, "shape_predictor_68_face_landmarks.dat")

# if not os.path.exists(PREDICTOR_PATH):
#     raise FileNotFoundError(f"File not found: {PREDICTOR_PATH}")

# predictor = dlib.shape_predictor(PREDICTOR_PATH)
# # Load face detector and eye landmark predictor
# detector = dlib.get_frontal_face_detector()
# # predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download required

# def eye_aspect_ratio(eye):
#     """Calculate eye aspect ratio to detect blinks"""
#     A = np.linalg.norm(eye[1] - eye[5])
#     B = np.linalg.norm(eye[2] - eye[4])
#     C = np.linalg.norm(eye[0] - eye[3])
#     return (A + B) / (2.0 * C)

# def detect_blink(frame, gray, face):
#     """Detect if a person blinks"""
#     shape = predictor(gray, face)
#     shape = face_utils.shape_to_np(shape)

#     left_eye = shape[42:48]
#     right_eye = shape[36:42]

#     left_EAR = eye_aspect_ratio(left_eye)
#     right_EAR = eye_aspect_ratio(right_eye)

#     ear = (left_EAR + right_EAR) / 2.0

#     return ear < 0.2  # If eyes closed, return True

# def mark_attendance(request):
#     video_capture = cv2.VideoCapture(0)

#     # Load employee face encodings
#     employees = Employee.objects.all()
#     known_face_encodings = []
#     known_face_names = []

#     for employee in employees:
#         try:
#             if employee.face_encoding:
#                 encoding_array = np.frombuffer(employee.face_encoding, dtype=np.float64)
#                 if encoding_array.size == 128:
#                     known_face_encodings.append(encoding_array)
#                     known_face_names.append(employee.name)
#                 else:
#                     print(f"Invalid encoding for {employee.name}")
#             else:
#                 print(f"No encoding for {employee.name}")
#         except Exception as e:
#             print(f"Error loading encoding for {employee.name}: {e}")

#     blink_detected = False  # Track if blink is detected

#     while True:
#         ret, frame = video_capture.read()
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Detect faces
#         faces = detector(gray)

#         for face in faces:
#             if detect_blink(frame, gray, face):
#                 blink_detected = True  # User blinked, allow attendance

#         # Find faces and encode
#         face_locations = face_recognition.face_locations(frame)
#         face_encodings = face_recognition.face_encodings(frame, face_locations)

#         for face_encoding, face_location in zip(face_encodings, face_locations):
#             matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
#             matches = [match.item() for match in matches]

#             name = "Unknown"

#             if True in matches and blink_detected:
#                 first_match_index = matches.index(True)
#                 name = known_face_names[first_match_index]
#                 try:
#                     attendance = Attendance.objects.create(user_id='001', time_in=timezone.now())
#                     print(f"Attendance marked at {attendance.time_in}")

#                 except Exception as e:
#                     print(f"Error marking attendance for {name}: {e}")

#             # Draw a rectangle
#             top, right, bottom, left = face_location
#             cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

#             # Label face
#             font = cv2.FONT_HERSHEY_DUPLEX
#             cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

#         # Show video
#         cv2.imshow('Video', frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     video_capture.release()
#     cv2.destroyAllWindows()
#     return JsonResponse({"message": "Attendance marked."})


# def mark_attendance(request):
#     # Capture video feed
#     video_capture = cv2.VideoCapture(0)
    
#     # Load all employee face encodings from the database
#     employees = Employee.objects.all()
#     known_face_encodings = []
#     known_face_names = []
    
#     for employee in employees:
#         try:
#             # Check if the employee has a valid face encoding
#             if employee.face_encoding:
#                 encoding_array = np.frombuffer(employee.face_encoding, dtype=np.float64)
#                 if encoding_array.size == 128:  # Ensure valid encoding size (128-dim)
#                     known_face_encodings.append(encoding_array)
#                     known_face_names.append(employee.name)
#                 else:
#                     print(f"Invalid encoding for {employee.name}")
#             else:
#                 print(f"No encoding for {employee.name}")
#         except Exception as e:
#             print(f"Error loading encoding for {employee.name}: {e}")
        
#     while True:
#         ret, frame = video_capture.read()

#         # Find faces in the frame
#         face_locations = face_recognition.face_locations(frame)
#         face_encodings = face_recognition.face_encodings(frame, face_locations)

#         for face_encoding, face_location in zip(face_encodings, face_locations):
#             matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

#             # Convert np.True_ to native Python True
#             matches = [match.item() for match in matches]
#             print("Known face names:", known_face_names)
            
#             name = "Unknown"

#             # If a match is found, assign the name
#             if True in matches:
#                 first_match_index = matches.index(True)
#                 name = known_face_names[first_match_index]

#                 # Mark attendance (store it in your database)
#                 try:
#                     # Get the user associated with the matched name
#                     # user = User.objects.get(username=name)
                    
#                     # Ensure user is authenticated (optional, since you're matching employees)
#                     # if user.is_authenticated:
#                     print("Matches:", matches)

#                         # Store attendance (mark time in)
#                         # attendance = Attendance.objects.create(user_id=user.id, time_in=timezone.now())
#                         # print(f"Attendance marked for {user.username} at {attendance.time_in}")
#                     # Store attendance (mark time in)
#                     attendance = Attendance.objects.create(user_id='001', time_in=timezone.now())
#                     print(f"Attendance marked at {attendance.time_in}")

#                 except User.DoesNotExist:
#                     name = "Unknown"  # If no user is found, keep name as "Unknown"

#             # Draw a rectangle around the face
#             top, right, bottom, left = face_location
#             cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

#             # Label the face with the name
#             font = cv2.FONT_HERSHEY_DUPLEX
#             cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

#         # Display the result
#         cv2.imshow('Video', frame)

#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     video_capture.release()
#     cv2.destroyAllWindows()
#     return JsonResponse({"message": "Attendance marked."})


# def mark_attendance(request):
#     # Capture video feed
#     video_capture = cv2.VideoCapture(0)
    
#     # Load all employee face encodings from the database
#     employees = Employee.objects.all()
#     known_face_encodings = []
#     known_face_names = []
    
#     for employee in employees:
#         # print(f"Encoding for {employee.name}: {employee.face_encoding}")

#         # If stored as a string of list, convert it back to numpy array
#         try:
#             encoding_list = eval(employee.face_encoding)  # Using eval() to convert string to list (use with caution)
#             known_face_encodings.append(np.array(encoding_list, dtype=np.float64))
#         except:
#             # Handle cases where face_encoding is not in correct format
#             print(f"Error with encoding for {employee.name}")
        
#         known_face_names.append(employee.name)
        
#     while True:
#         ret, frame = video_capture.read()

#         # Find faces in the frame
#         face_locations = face_recognition.face_locations(frame)
#         face_encodings = face_recognition.face_encodings(frame, face_locations)

#         for face_encoding, face_location in zip(face_encodings, face_locations):
#             matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
#             print("Known face names:", known_face_names)
#             print("Known face encodings:", known_face_encodings)
#             print("Matches:", matches)
#             name = "Unknown"

#             # If a match is found, assign the name
#             if True in matches:
#                 first_match_index = matches.index(True)
#                 name = known_face_names[first_match_index]

#                 # Mark attendance (store it in your database)
#                 try:
#                     # Get the user associated with the matched name
#                     user = User.objects.get(username=name)
                    
#                     # Ensure user is authenticated (optional, since you're matching employees)
#                     if user.is_authenticated:
#                         # Store attendance (mark time in)
#                         Attendance.objects.create(user=user, time_in=timezone.now())

#                 except User.DoesNotExist:
#                     name = "Unknown"  # If no user is found, keep name as "Unknown"

#             # Draw a rectangle around the face
#             top, right, bottom, left = face_location
#             cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

#             # Label the face with the name
#             font = cv2.FONT_HERSHEY_DUPLEX
#             cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

#         # Display the result
#         cv2.imshow('Video', frame)

#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     video_capture.release()
#     cv2.destroyAllWindows()
#     return JsonResponse({"message": "Attendance marked."})

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
