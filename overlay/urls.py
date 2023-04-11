from django.urls import path
from . import views

urlpatterns = [
    path("overlay/", views.overlay_jewellery, name="overlay_jewellery")
]