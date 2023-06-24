from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
import cv2
import numpy as np
import dlib
import os
import requests
import io
import pickle

# Create your views here.
@csrf_exempt
@api_view(['POST'])

def overlay_jewellery(request):
    try:    
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
            jewellery_height = int(abs(neck_point.y - chin_point.y)) + 125
            jewellery_width = int(jewellery_height * jewellery_img.shape[1] / jewellery_img.shape[0]) + 30

            # Resize the jewellery to fit the neck region
            resized_jewellery = cv2.resize(jewellery_img, (jewellery_width, jewellery_height))

            # Calculate the position of the jewellery
            jewellery_x = int(chin_point.x - jewellery_width / 2) + 5
            jewellery_y = int(chin_point.y) + 10

            # Overlay the jewellery on the neck region
            for i in range(jewellery_height):
                for j in range(jewellery_width):
                    if resized_jewellery[i, j][3] != 0:
                        user_face[jewellery_y + i, jewellery_x + j] = resized_jewellery[i, j][:3]

        # Resize the frame to match the output window size
        height, width = user_face.shape[:2]
        user_face = cv2.resize(user_face, (int(width / 2), int(height / 2)))

        # Convert the image to JPEG format in memory
        _, image_buffer = cv2.imencode('.jpg', user_face)
        
        # Create an in-memory byte stream
        image_stream = io.BytesIO(image_buffer.tobytes())

        # Return the image as a response
        return HttpResponse(image_stream, content_type='image/jpeg')

    except Exception as e:
        if type(e).__name__ == "IndexError":
            return HttpResponse("Try again.")
        #return HttpResponse(f"Error : {type(e).__name__}")

#-----------------------------------------------------------------------------------------------------------------

@csrf_exempt
@api_view(['POST'])

def overlay_earrings(request):
    try:    
        # Get the user's face image and earrings image from the request data
        user_face_link = request.data.get('user_face')
        earrings_img_link = request.data.get('earrings')

        # Download the user's face image from Firebase
        user_face_response = requests.get(user_face_link)
        user_face_data = user_face_response.content

        # Download the earrings image from Firebase
        earrings_img_response = requests.get(earrings_img_link)
        earrings_img_data = earrings_img_response.content

        # Decode the image data using OpenCV
        user_face = cv2.imdecode(np.frombuffer(user_face_data, np.uint8), -1)
        earrings_img = cv2.imdecode(np.frombuffer(earrings_img_data, np.uint8), -1)

        # Initialize the face detector and landmark predictor
        detector = dlib.get_frontal_face_detector()
        predictor_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shape_predictor_68_face_landmarks.dat')
        predictor = dlib.shape_predictor(predictor_path)

    #-----------------------------------------------------   

        def overlay_earring(image, earring_image, landmarks, start_point, end_point):
            # Calculate the size of the earring
            earring_height = int(abs(end_point.y - start_point.y)) + 30
            earring_width = int(earring_height * earring_image.shape[1] / earring_image.shape[0]) + 5

            # Resize the earring to fit the ear region
            resized_earring = cv2.resize(earring_image, (earring_width, earring_height))

            # Calculate the position of the earring
            earring_x = int(start_point.x - earring_width / 2)
            earring_y = int(start_point.y - earring_height / 2) + 40

            # Overlay the earring on the ear region
            for i in range(earring_height):
                for j in range(earring_width):
                    if resized_earring[i, j][3] != 0:
                        image[earring_y + i, earring_x + j] = resized_earring[i, j][:3]

            return image
        
    #-------------------------------------------------------
        
        # Detect faces in the user's face image
        faces = detector(user_face, 0)
        
        # Loop through each face and overlay the earrings
        for face in faces:
            # Detect facial landmarks
            landmarks = predictor(user_face, face)

            # Get the coordinates of the ear landmarks
            left_ear_start = landmarks.part(1)
            left_ear_end = landmarks.part(2)
            right_ear_start = landmarks.part(15)
            right_ear_end = landmarks.part(14)

            # Overlay earrings on each ear
            user_face = overlay_earring(user_face, earrings_img, landmarks, left_ear_start, left_ear_end)
            user_face = overlay_earring(user_face, earrings_img, landmarks, right_ear_start, right_ear_end)

        # Resize the frame to match the output window size
        height, width = user_face.shape[:2]
        user_face = cv2.resize(user_face, (int(width / 2), int(height / 2)))

        # Convert the image to JPEG format in memory
        _, image_buffer = cv2.imencode('.jpg', user_face)
        
        # Create an in-memory byte stream
        image_stream = io.BytesIO(image_buffer.tobytes())

        # Return the image as a response
        return HttpResponse(image_stream, content_type='image/jpeg')
    
    except Exception as e:
        return HttpResponse(f"Error : {type(e).__name__}")

#-----------------------------------------------------------------------------------------------------------------
  
@csrf_exempt
@api_view(['POST'])

def face_shape(request):
    try:
        # Get the user's face image and earrings image from the request data
        user_face_link = request.data.get('user_face')

        # Download the user's face image from Firebase
        user_face_response = requests.get(user_face_link)
        user_face = user_face_response.content

        # Load the trained model
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pkl')
        with open(model_path, "rb") as file:
            model = pickle.load(file)
            
        IMG_SIZE = (100, 100)
        CATEGORIES = ["heart", "oblong", "oval", "round", "square"]

        # Load and preprocess the new image
        nparr = np.frombuffer(user_face, np.uint8)
        new_image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        new_image = cv2.resize(new_image, IMG_SIZE)
        new_X = np.array(new_image).reshape(1, -1)

        # Predict the face shape
        predicted_label = model.predict(new_X)
        predicted_face_shape = CATEGORIES[predicted_label[0]]
        
        # Return the response
        return HttpResponse(str(predicted_face_shape))

    except Exception as e:
        return HttpResponse(f"Error : {type(e).__name__}")