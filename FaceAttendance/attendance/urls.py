from django.urls import path
from django.conf.urls.static import static
from django.conf import settings
from .views import attendance_view, video_feed, live_feed_page
from . import views


urlpatterns = [
    path('attendance', attendance_view, name='attendance'),  # /attendance/ route
    # path('mark_attendance/', views.mark_attendance, name='mark_attendance'),
    path('generate-face-encodings/', views.generate_and_store_face_encoding, name='generate_face_encodings'),
    path('live_feed/', live_feed_page, name='live_feed'),
    path('video_feed/', video_feed, name='video_feed'),
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)