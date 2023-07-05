import json
from rest_framework.views import APIView
from .serializer import *
from rest_framework.permissions import AllowAny
from rest_framework.parsers import MultiPartParser
from django.http import JsonResponse
import face_recognition
import os
import numpy as np
from rest_framework.parsers import MultiPartParser
import pandas as pd
import tensorflow as tf
from django.http import JsonResponse
from PIL import Image



def getFace(img):
    image = face_recognition.load_image_file(img)
  
    return image


def findEcoding(img):
    encodings = []
    names = []

    image = getFace(img)

    face_locations = face_recognition.face_locations(image, model="hog")
    face_detection = face_recognition.face_encodings(image,known_face_locations=face_locations)
   

    for faceing in face_detection:
        faceing = np.array(faceing).ravel()
        encodings.append(faceing)
        names.append(os.path.basename(img).split('.')[0])

    return encodings,names

def load_labels_from_file(filename):
    with open(filename, 'r') as file:
        labels = file.read().splitlines()
    return labels

class predict_image(APIView):
    permission_classes = [AllowAny]
    parser_classes = [MultiPartParser]
    def post(self,request):
        if request.method == 'POST':
            image_data = request.FILES.get('image')
            image_instance = image(image=image_data)
            image_instance.save()
            my_map = dict()
            model = tf.keras.models.load_model('D:\\college\\final project\\back new version\\myfinalproject\\mybackend\\savedModel\\my_model.h5')
            my_image, _ = findEcoding(image_instance.image.path)
            my_image = np.array(my_image)
            my_image = np.expand_dims(my_image, axis=0)
            my_image = my_image.reshape(-1, 128, 1) 
            names = load_labels_from_file("D:\\college\\final project\\back new version\\myfinalproject\\mybackend\\savedModel\\Labels")
            nums = load_labels_from_file("D:\\college\\final project\\back new version\\myfinalproject\\mybackend\\savedModel\\Labels2")
            for i in range(len(nums)):
                 my_map[int(nums[i])] = names[i]
            predictions = model.predict(my_image)
            for i in range(len(predictions)):
                predicted_label = ([np.argmax(predictions[i])])
                print(my_map[predicted_label[0]])
            image_instance.classification_results = predictions 
            image_instance.save()
            merged_list = [my_map[np.argmax(prediction)] for prediction in predictions]
            response_data = {'prediction': merged_list}
            return JsonResponse(response_data)
        else:
            return JsonResponse({'error': 'Invalid request method'})


