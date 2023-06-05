from django.urls import path
from . import views

urlpatterns = [
    path("overlay_jewellery/", views.overlay_jewellery, name="overlay_jewellery"),
    path("overlay_earrings/", views.overlay_earrings, name="overlay_earrings")
]