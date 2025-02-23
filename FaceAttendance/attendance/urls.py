from django.urls import path
from .views_1 import attendance_view, video_feed, live_feed_page
from . import views_1

urlpatterns = [
    path('attendance', attendance_view, name='attendance'),  # /attendance/ route
    # path('mark_attendance/', views.mark_attendance, name='mark_attendance'),
    path('generate-face-encodings/', views_1.generate_and_store_face_encoding, name='generate_face_encodings'),

    path('video_feed/', video_feed, name='video_feed'),
    path('live_feed/', live_feed_page, name='live_feed'),
]

192.168.1.233