from django.urls import path
from .views import attendance_view
from . import views

urlpatterns = [
    path('attendance', attendance_view, name='attendance'),  # /attendance/ route
    path('mark_attendance/', views.mark_attendance, name='mark_attendance'),
    path('generate-face-encodings/', views.generate_and_store_face_encoding, name='generate_face_encodings'),
]