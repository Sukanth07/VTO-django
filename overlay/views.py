from django.http import HttpResponse
from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
import cv2
import numpy as np
import dlib
import os

current_directory = os.path.dirname(os.path.abspath(__file__))

#user face
#user_face_path = os.path.join(current_directory, 'user_face2.jpg')
#user_face = cv2.imread(user_face_path, -1)

# Load the jewellery image
jewellery_img_path = os.path.join(current_directory, 'jewellery8.png')
jewellery_img = cv2.imread(jewellery_img_path, -1)

# Initialize the face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor_path = os.path.join(current_directory, 'shape_predictor_68_face_landmarks.dat')
predictor = dlib.shape_predictor(predictor_path)

# Create your views here.
@api_view(['POST'])

def overlay_jewellery(request):
    
    #frame_bytes = request.data.get('frame')
    #nparr = np.frombuffer(frame_bytes.read(), np.uint8)
    #frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    user_face = request.data.get('url')
    

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

    retval, buffer = cv2.imencode('.jpg', frame)
    response = HttpResponse(buffer.tobytes(), content_type='image/jpeg')

    return response     

    # Generate a unique file name for the image
    #file_name = 'output_image.jpg'

    # Build the full file path
    #file_path = os.path.join(current_directory, file_name)

    # Save the image frame to a file
    #cv2.imwrite(file_path, frame)
    
    #return
    
