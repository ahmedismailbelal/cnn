from django.conf import settings
from django.conf.urls.static import static
from django.urls import path, include
from .views import *

urlpatterns = [
    path ('image', predict_image.as_view() , name='image'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
