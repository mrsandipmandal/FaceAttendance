from django import forms
from django.contrib import admin
from .models import Employee, Attendance

class EmployeeAdminForm(forms.ModelForm):
    CAPTURE_CHOICES = [
        ('file', 'Upload Image'),
        ('webcam', 'Capture from Webcam')
    ]
    
    capture_method = forms.ChoiceField(
        choices=CAPTURE_CHOICES, 
        widget=forms.RadioSelect, 
        initial='file',
        required=False
    )

    class Meta:
        model = Employee
        fields = ['name', 'emp_id', 'image']
        widgets = {
            'image': forms.FileInput(attrs={'accept': 'image/*'})
        }

@admin.register(Employee)
class EmployeeAdmin(admin.ModelAdmin):
    form = EmployeeAdminForm
    list_display = ('name', 'emp_id')

    class Media:
        js = ('admin/js/webcam_capture.js',)




@admin.register(Attendance)
class AttendanceAdmin(admin.ModelAdmin):
    list_display = ('employee', 'date', 'time_in')
    list_filter = ('date', 'employee')
