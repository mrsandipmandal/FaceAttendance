from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone  # 
from django.db import models
import face_recognition
import numpy as np
from django.core.exceptions import ValidationError


# for single employee image upload
# class Employee(models.Model):
#     name = models.CharField(max_length=100)
#     emp_id = models.CharField(max_length=50, unique=True)
#     image = models.ImageField(upload_to='employees/')
#     face_encoding = models.BinaryField(blank=True, null=True)  # Store the binary data of the encoding
#     def save_face_encoding(self):
#         try:
#             # Print the image path to check if it resolves correctly
#             print("Image path:", self.image.path)
            
#             # Load the image file from the employee's image path
#             image_path = self.image.path
#             image = face_recognition.load_image_file(image_path)

#             # Get face encodings
#             face_encodings = face_recognition.face_encodings(image)

#             if face_encodings:
#                 encoding = face_encodings[0]
#                 encoding_array = np.array(encoding, dtype=np.float64)
#                 self.face_encoding = encoding_array.tobytes()
#                 self.save()
#             else:
#                 raise ValidationError(f"No face found in the image for employee {self.name}")

#         except Exception as e:
#             raise ValidationError(f"Error while saving face encoding for {self.name}: {str(e)}")

# for multiple employee image upload
class Employee(models.Model):
    name = models.CharField(max_length=100)
    emp_id = models.CharField(max_length=50, unique=True)
    image = models.ImageField(upload_to='employees/')
    image2 = models.ImageField(upload_to='employees/', null=True, blank=True)
    image3 = models.ImageField(upload_to='employees/', null=True, blank=True)
    image4 = models.ImageField(upload_to='employees/', null=True, blank=True)
    image5 = models.ImageField(upload_to='employees/', null=True, blank=True)
    face_encoding = models.BinaryField(blank=True, null=True)

    def save_face_encoding(self):
        try:
            # Get all available images
            image_paths = [
                self.image.path,
                self.image2.path if self.image2 else None,
                self.image3.path if self.image3 else None,
                self.image4.path if self.image4 else None,
                self.image5.path if self.image5 else None
            ]
            
            # Filter out None values
            valid_paths = [path for path in image_paths if path]
            
            all_encodings = []
            for path in valid_paths:
                image = face_recognition.load_image_file(path)
                face_encodings = face_recognition.face_encodings(image)
                if face_encodings:
                    all_encodings.append(face_encodings[0])

            if all_encodings:
                # Calculate average encoding
                average_encoding = np.mean(all_encodings, axis=0)
                self.face_encoding = average_encoding.tobytes()
                self.save()
            else:
                raise ValidationError(f"No faces found in any images for employee {self.name}")

        except Exception as e:
            raise ValidationError(f"Error while saving face encoding for {self.name}: {str(e)}")


class Attendance(models.Model):
    employee = models.ForeignKey(Employee, on_delete=models.CASCADE)  # ✅ Use ForeignKey instead of IntegerField
    time_in = models.DateTimeField()
    time_out = models.DateTimeField(null=True, blank=True)
    date = models.DateField(default=timezone.now)
    image_path = models.CharField(max_length=255, null=True, blank=True)

    def __str__(self):
        return f"{self.employee.name} - {self.time_in}"

