from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

admin.site.site_header = 'Face Attendance Admin'
admin.site.site_title = 'Face Attendance Portal'
admin.site.index_title = 'Welcome to Face Attendance Admin'

urlpatterns = [
    path('admin/', admin.site.urls),
    path('attendance/', include('attendance.urls')),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
