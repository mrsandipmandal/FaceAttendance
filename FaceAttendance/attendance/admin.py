from django.contrib import admin
from .models import Employee  # Import the Employee model
from .models import Attendance

admin.site.register(Employee)  # Register the model
admin.site.register(Attendance)
