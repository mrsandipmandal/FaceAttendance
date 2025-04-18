# Capture Live Video for Attendance
import cv2
import face_recognition
import numpy as np
import dlib
import pyttsx3
import threading
import queue
from imutils import face_utils
from django.http import StreamingHttpResponse, JsonResponse
from django.utils import timezone
from django.shortcuts import render
from .models import Employee, Attendance
import os
import logging



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTOR_PATH = os.path.join(BASE_DIR, "shape_predictor_68_face_landmarks.dat")

# Face Recognition Configuration
FACE_RECOGNITION_TOLERANCE = 0.4  # Adjustable tolerance
MIN_FACE_MATCHES = 1  # Minimum number of face matches required

class FaceAttendanceSystem:
    def __init__(self):
        # Initialize face recognition components
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(PREDICTOR_PATH)
        
        # Speech synthesis setup
        self.speech_queue = queue.Queue()
        self.speech_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.speech_thread.start()

        # Face recognition data
        self.known_face_encodings = []
        self.known_face_names = []
        self.employee_ids = []

    def _speech_worker(self):
        """Background thread for speech synthesis"""
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        
        while True:
            text = self.speech_queue.get()
            if text is None:
                break
            
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                logger.error(f"Speech synthesis error: {e}")
            
            self.speech_queue.task_done()

    def speak(self, text):
        """Add speech request to queue"""
        self.speech_queue.put(text)

    def preprocess_frame(self, frame):
        """Advanced frame preprocessing"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        return enhanced

    def load_known_faces(self):
        """Load known face encodings from database"""
        employees = Employee.objects.all()
        self.known_face_encodings.clear()
        self.known_face_names.clear()
        self.employee_ids.clear()

        for employee in employees:
            if employee.face_encoding:
                try:
                    encoding = np.frombuffer(employee.face_encoding, dtype=np.float64)
                    if encoding.size == 128:
                        self.known_face_encodings.append(encoding)
                        self.known_face_names.append(employee.name)
                        self.employee_ids.append(employee.id)
                except Exception as e:
                    logger.error(f"Error loading face encoding for {employee.name}: {e}")

# single face recognition
    # def recognize_faces(self, frame):
    #     """Advanced face recognition method"""
    #     # Preprocess frame
    #     preprocessed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
    #     # Detect faces
    #     face_locations = face_recognition.face_locations(preprocessed_frame)
    #     face_encodings = face_recognition.face_encodings(preprocessed_frame, face_locations)

    #     recognized_faces = []

    #     for face_encoding, face_location in zip(face_encodings, face_locations):
    #         # Compare faces with custom tolerance
    #         matches = face_recognition.compare_faces(
    #             self.known_face_encodings, 
    #             face_encoding, 
    #             tolerance=FACE_RECOGNITION_TOLERANCE
    #         )

    #         # Compute face distances
    #         face_distances = face_recognition.face_distance(
    #             self.known_face_encodings, 
    #             face_encoding
    #         )

    #         # Find best match
    #         best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else -1

    #         if best_match_index != -1 and matches[best_match_index]:
    #             name = self.known_face_names[best_match_index]
    #             employee_id = self.employee_ids[best_match_index]
    #             recognized_faces.append((name, employee_id, face_location))

    #     return recognized_faces

# multiple face recognition
    def recognize_faces(self, frame):
        """Enhanced face recognition with multiple angle matching"""
        preprocessed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces with different angles
        face_locations = face_recognition.face_locations(preprocessed_frame, model="cnn")
        face_encodings = face_recognition.face_encodings(preprocessed_frame, face_locations)

        recognized_faces = []

        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Use stricter matching criteria
            matches = face_recognition.compare_faces(
                self.known_face_encodings, 
                face_encoding, 
                tolerance=FACE_RECOGNITION_TOLERANCE
            )
            
            face_distances = face_recognition.face_distance(
                self.known_face_encodings, 
                face_encoding
            )

            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                confidence = 1 - face_distances[best_match_index]
                
                # Only accept matches with high confidence
                if matches[best_match_index] and confidence > 0.6:
                    name = self.known_face_names[best_match_index]
                    employee_id = self.employee_ids[best_match_index]
                    recognized_faces.append((name, employee_id, face_location))

        return recognized_faces


    def mark_attendance(self, employee_id, frame, face_location):
        """Mark attendance for recognized employee and save face image"""
        try:
            # Modify the save path to be at the root of the project
            # save_path = os.path.join(BASE_DIR, "media", "attendance_images") # old path 
            save_path = os.path.join("media", "attendance_images") # use root media folder
            os.makedirs(save_path, exist_ok=True)

            # Extract face region
            top, right, bottom, left = face_location
            face_image = frame[top:bottom, left:right]

            # Define image filename
            filename = f"{employee_id}_{timezone.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            image_path = os.path.join(save_path, filename)

            # Save the face image
            cv2.imwrite(image_path, face_image)

            # Fetch the Employee instance
            employee = Employee.objects.get(id=employee_id)  
            print(f"Employee: {employee}")

            # Check if attendance already exists
            existing_attendance = Attendance.objects.filter(
                employee=employee,
                date=timezone.now().date(),
                time_in__isnull=False
            ).first()
        
            if existing_attendance:
                logger.info(f"Attendance already marked for employee ID: {employee_id}")
                return None

            # Save attendance with image path
            attendance = Attendance.objects.create(
                employee=employee,
                time_in=timezone.now(),
                date=timezone.now().date(),
                image_path=f"attendance_images/{filename}"  # Store relative path
            )
        
            logger.info(f"Attendance marked for employee ID: {employee_id}, Image saved: {filename}")
            return attendance

        except Exception as e:
            logger.error(f"Attendance marking error: {e}")
            return None


# Global instance of face attendance system
face_attendance = FaceAttendanceSystem()


""" for webcam video capture """
def generate_frames(request):
    """Generate video frames with face recognition"""
    # Reload known faces before starting
    face_attendance.load_known_faces()
    
    video_capture = cv2.VideoCapture(0)
    recognized_faces = set()

    while True:
        success, frame = video_capture.read()
        if not success:
            break

        # Recognize faces
        detected_faces = face_attendance.recognize_faces(frame)

        for name, employee_id, face_location in detected_faces:
            top, right, bottom, left = face_location
            
            # Draw rectangle and name
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

            # Mark attendance and speak only for new recognitions
            if name not in recognized_faces:
                recognized_faces.add(name)
                face_attendance.speak(f"Welcome {name}")
                # face_attendance.mark_attendance(employee_id) # old code for marking attendance
                face_attendance.mark_attendance(employee_id, frame, face_location)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    video_capture.release()

""" for cctv video capture """
# def generate_frames(request):
#     face_attendance.load_known_faces()
    
#     DVR_URL = "rtsp://admin:123456@192.168.1.233:554/Streaming/Channels/101"
#     video_capture = cv2.VideoCapture(DVR_URL)

#     if not video_capture.isOpened():
#         logger.error("Could not open DVR stream")
#         return

#     recognized_faces = set()

#     while True:
#         success, frame = video_capture.read()
#         if not success:
#             break

#         detected_faces = face_attendance.recognize_faces(frame)

#         for name, employee_id, face_location in detected_faces:
#             top, right, bottom, left = face_location
#             cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
#             cv2.putText(frame, name, (left + 6, bottom - 6),
#                         cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

#             if name not in recognized_faces:
#                 recognized_faces.add(name)
#                 face_attendance.speak(f"Welcome {name}")
#                 face_attendance.mark_attendance(employee_id)

#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()
        
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#     video_capture.release()


def video_feed(request):
    return StreamingHttpResponse(
        generate_frames(request), 
        content_type='multipart/x-mixed-replace; boundary=frame'
    )

def live_feed_page(request):
    """Render live feed page"""
    return render(request, 'attendance/live_feed.html')

# single face encoding
# def generate_and_store_face_encoding(request):
#     employees = Employee.objects.all()

#     for employee in employees:
#         try:
#             # Assuming 'image' is the file field of Employee model
#             image_path = employee.image.path
#             # Load the image from the file path
#             image = face_recognition.load_image_file(image_path)

#             # Generate face encodings for the image (assuming one face per image)
#             face_encodings = face_recognition.face_encodings(image)

#             if face_encodings:
#                 # Convert the face encoding into bytes to store in the database
#                 face_encoding_bytes = face_encodings[0].tobytes()

#                 # Update employee with the generated face encoding
#                 employee.face_encoding = face_encoding_bytes
#                 employee.save()

#                 print(f"Face encoding saved for {employee.name}")
#             else:
#                 print(f"No face found for {employee.name}")

#         except Exception as e:
#             print(f"Error processing image for {employee.name}: {e}")

#     return JsonResponse({"message": "Face encodings generated and stored."})

# multiple face encoding
def generate_and_store_face_encoding(request):
    employees = Employee.objects.all()

    for employee in employees:
        try:
            # Get all 5 training images for each employee
            image_paths = [
                employee.image.path,  # Main image
                employee.image2.path, # Add these fields to Employee model
                employee.image3.path,
                employee.image4.path, 
                employee.image5.path
            ]
            
            all_encodings = []
            for image_path in image_paths:
                image = face_recognition.load_image_file(image_path)
                face_encodings = face_recognition.face_encodings(image)
                
                if face_encodings:
                    all_encodings.append(face_encodings[0])

            if all_encodings:
                # Store average encoding of all 5 images
                average_encoding = np.mean(all_encodings, axis=0)
                face_encoding_bytes = average_encoding.tobytes()
                
                employee.face_encoding = face_encoding_bytes
                employee.save()
                print(f"Multiple face encodings averaged and saved for {employee.name}")

        except Exception as e:
            print(f"Error processing images for {employee.name}: {e}")

    return JsonResponse({"message": "Multi-image face encodings generated and stored."})


def attendance_view(request):
    return render(request, 'attendance/attendance.html')