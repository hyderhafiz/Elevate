from django.urls import path
from . import views

urlpatterns = [
      path('detection/', views.detection_system, name='detection_system'),
]
