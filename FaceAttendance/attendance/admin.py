# admin.py
from django.contrib import admin
from django.utils.html import mark_safe
from django.conf import settings
from .models import Employee, Attendance
import os


# admin.site.register(Employee)
@admin.register(Employee)
class EmployeeAdmin(admin.ModelAdmin):
    # Columns to display in the list view
    list_display = (
        'id', 
        'name', 
        'emp_id', 
        'display_image', 
    )
    
    # Columns that can be used to search
    search_fields = ('name', 'emp_id')
    
    # Filters on the right side of the admin page
    list_filter = ('id',)
    
    # Custom method to display image thumbnail
    def display_image(self, obj):
        # print(f"EmployeeAdmin : {obj.image}")
        if obj.image:
            return mark_safe(f'<a href="{obj.image.url}" target="_blank"><img src="{obj.image.url}" width="100" height="100" /></a>')
        return 'No Image'
    
    display_image.short_description = 'Image'
    display_image.allow_tags = True

    # Customize the form
    fieldsets = (
        ('Personal Information', {
            'fields': ('name', 'emp_id', 'image')
        }),
        # Add more fieldsets as needed
    )

    # Number of items per page
    list_per_page = 20

    # Actions that can be performed on multiple selected rows
    actions = ['generate_face_encoding']

    def generate_face_encoding(self, request, queryset):
        for employee in queryset:
            try:
                employee.save_face_encoding()
            except Exception as e:
                self.message_user(request, f"Error generating face encoding for {employee.name}: {str(e)}")
    generate_face_encoding.short_description = "Generate Face Encoding"



# admin.site.register(Attendance)
@admin.register(Attendance)
class AttendanceAdmin(admin.ModelAdmin):
    list_display = (
        'id',
        'employee__name',
        'date',
        'time_in',
        'time_out',
        'display_image'
    )
    
    # Columns that can be used to search
    search_fields = ('date', 'employee__name', 'employee__emp_id')
    
    # Filters on the right side of the admin page
    list_filter = ('id',)
    
    def display_image(self, obj):
        if obj.image_path:
            return mark_safe(f'<a href="{settings.MEDIA_URL}{obj.image_path}" target="_blank"><img src="{settings.MEDIA_URL}{obj.image_path}" width="100" height="100" /></a>')
        return 'No Image'

    display_image.short_description = 'Image'

    # Customize the form
    fieldsets = (
        ('Attendance Information', {
            'fields': ('employee', 'date', 'time_in', 'time_out', 'image_path')
        }),
    )

    list_per_page = 20