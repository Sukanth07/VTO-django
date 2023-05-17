from django.http import HttpResponse
from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
import cv2
import numpy as np
import dlib
from django.views.decorators.csrf import csrf_exempt
#import os

#current_directory = os.path.dirname(os.path.abspath(__file__))

# Initialize the face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
#predictor_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shape_predictor_68_face_landmarks.dat')
predictor_path = 'VTO_DJANGO/overlay/shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)



# Create your views here.
@csrf_exempt
@api_view(['POST'])

def overlay_jewellery(request):
    
    user_face = request.POST.get('user_face')
    jewellery_img = request.POST.get('jewellery')
    

    frame = user_face
    
    faces = detector(frame, 0)

    # Loop through each face and overlay the jewellery
    for face in faces:
        # Detect facial landmarks
        landmarks = predictor(frame, face)

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
                if resized_jewellery[i,j][3] != 0:
                    frame[jewellery_y+i, jewellery_x+j] = resized_jewellery[i,j][:3]
        
    # Resize the frame to match the output window size
    height, width = frame.shape[:2]
    frame = cv2.resize(frame, (int(width/2), int(height/2)))
    
    # Generate a unique file name for the image
    file_name = 'output_image.jpg'

    # Build the full file path
    #file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)

    file_path = 'VTO_DJANGO/overlay/' + file_name
    
    # Save the image frame to a file
    cv2.imwrite(file_path, frame)

    return HttpResponse("Image is ready to return")
    # Return the final image as a response
    #with open(file_path, 'rb') as image_file:
    #    return HttpResponse(image_file.read(), content_type='image/jpeg')  

    
