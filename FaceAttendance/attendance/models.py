from django.db import models
from django.contrib.auth.models import User


from django.db import models
import face_recognition
import numpy as np
from django.core.exceptions import ValidationError

class Employee(models.Model):
    name = models.CharField(max_length=100)
    emp_id = models.CharField(max_length=50, unique=True)
    image = models.ImageField(upload_to='employees/')
    face_encoding = models.BinaryField(blank=True, null=True)  # Store the binary data of the encoding
    def save_face_encoding(self):
        try:
            # Print the image path to check if it resolves correctly
            print("Image path:", self.image.path)
            
            # Load the image file from the employee's image path
            image_path = self.image.path
            image = face_recognition.load_image_file(image_path)

            # Get face encodings
            face_encodings = face_recognition.face_encodings(image)

            if face_encodings:
                encoding = face_encodings[0]
                encoding_array = np.array(encoding, dtype=np.float64)
                self.face_encoding = encoding_array.tobytes()
                self.save()
            else:
                raise ValidationError(f"No face found in the image for employee {self.name}")

        except Exception as e:
            raise ValidationError(f"Error while saving face encoding for {self.name}: {str(e)}")



class Attendance(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    time_in = models.DateTimeField()
    time_out = models.DateTimeField(null=True, blank=True)
    
    def __str__(self):
        return f"{self.user.username} - {self.time_in}"
    class Meta:
        ordering = ['-time_in']