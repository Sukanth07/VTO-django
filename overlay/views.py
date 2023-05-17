from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from rest_framework.response import Response
import cv2
import numpy as np
import dlib
import os
import base64
import requests
import io

# Create your views here.
@csrf_exempt
@api_view(['POST'])
def overlay_jewellery(request):
    # Get the user's face image and jewellery image from the request data
    user_face_link = request.data.get('user_face')
    jewellery_img_link = request.data.get('jewellery')

    # Download the user's face image from Firebase
    user_face_response = requests.get(user_face_link)
    user_face_data = user_face_response.content
    
    # Download the jewellery image from Firebase
    jewellery_img_response = requests.get(jewellery_img_link)
    jewellery_img_data = jewellery_img_response.content
    
    # Decode the image data using OpenCV
    user_face = cv2.imdecode(np.frombuffer(user_face_data, np.uint8), -1)
    jewellery_img = cv2.imdecode(np.frombuffer(jewellery_img_data, np.uint8), -1)
    
    # Initialize the face detector and landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shape_predictor_68_face_landmarks.dat')
    predictor = dlib.shape_predictor(predictor_path)

    # Detect faces in the user's face image
    faces = detector(user_face, 0)

    # Loop through each face and overlay the jewellery
    for face in faces:
        # Detect facial landmarks
        landmarks = predictor(user_face, face)

        chin_point = landmarks.part(8)
        neck_point = landmarks.part(5)

        # Calculate the size of the jewellery
        jewellery_height = int(abs(neck_point.y - chin_point.y)) + 110
        jewellery_width = int(jewellery_height * jewellery_img.shape[1] / jewellery_img.shape[0]) + 25

        # Resize the jewellery to fit the neck region
        resized_jewellery = cv2.resize(jewellery_img, (jewellery_width, jewellery_height))

        # Calculate the position of the jewellery
        jewellery_x = int(chin_point.x - jewellery_width / 2)
        jewellery_y = int(chin_point.y) + 15

        # Overlay the jewellery on the neck region
        for i in range(jewellery_height):
            for j in range(jewellery_width):
                if resized_jewellery[i, j][3] != 0:
                    user_face[jewellery_y + i, jewellery_x + j] = resized_jewellery[i, j][:3]

    # Resize the frame to match the output window size
    height, width = user_face.shape[:2]
    user_face = cv2.resize(user_face, (int(width / 2), int(height / 2)))

    # Generate a unique file name for the image
    file_name = 'output_image.jpg'

    # Build the full file path
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)

    # Save the image frame to a file
    cv2.imwrite(file_path, user_face)

    #return HttpResponse('Image ready to return')
    # Return the final image as a response
    with open(file_path, 'rb') as image_file:
        return HttpResponse(image_file.read(), content_type='image/jpeg')
