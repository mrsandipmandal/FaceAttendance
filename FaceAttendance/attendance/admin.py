# admin.py
from django.contrib import admin
from django.utils.html import mark_safe
from django.conf import settings
from .models import Employee, Attendance
import os


# admin.site.register(Employee)
@admin.register(Employee)
class EmployeeAdmin(admin.ModelAdmin):
    list_display = (
        'id', 
        'name', 
        'emp_id', 
        'display_image',
        'display_image2',
        'display_image3',
        'display_image4',
        'display_image5',
    )
    
    search_fields = ('name', 'emp_id')
    list_filter = ('id',)
    
    def display_image(self, obj):
        if obj.image:
            return mark_safe(f'<a href="{obj.image.url}" target="_blank"><img src="{obj.image.url}" width="100" height="100" /></a>')
        return 'No Image'
    
    display_image.short_description = 'Image'
    display_image.allow_tags = True    
    
    def display_image2(self, obj):
        if obj.image2:
            return mark_safe(f'<a href="{obj.image2.url}" target="_blank"><img src="{obj.image2.url}" width="100" height="100" /></a>')
        return 'No Image'
    
    display_image2.short_description = 'Image2'
    display_image2.allow_tags = True
    
    def display_image3(self, obj):
        if obj.image3:
            return mark_safe(f'<a href="{obj.image3.url}" target="_blank"><img src="{obj.image3.url}" width="100" height="100" /></a>')
        return 'No Image'
    
    display_image3.short_description = 'Image3'
    display_image3.allow_tags = True    
    
    def display_image4(self, obj):
        if obj.image4:
            return mark_safe(f'<a href="{obj.image4.url}" target="_blank"><img src="{obj.image4.url}" width="100" height="100" /></a>')
        return 'No Image'
    
    display_image4.short_description = 'Image4'
    display_image4.allow_tags = True
    
    def display_image5(self, obj):
        if obj.image5:
            return mark_safe(f'<a href="{obj.image5.url}" target="_blank"><img src="{obj.image5.url}" width="100" height="100" /></a>')
        return 'No Image'
    
    display_image5.short_description = 'Image5'
    display_image5.allow_tags = True

    fieldsets = (
        ('Personal Information', {
            'fields': ('name', 'emp_id')
        }),
        ('Images', {
            'fields': ('image', 'image2', 'image3', 'image4', 'image5')
        }),
    )

    list_per_page = 20
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