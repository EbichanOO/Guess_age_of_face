import cv2
import datetime
import os
import numpy as np
from face_sex_play import face_ageer

spsp = True

print('plz write your name.')
fileName = input('>>> ')
#print(fileName)

cascade_path = "C:/Users/b1017089/Desktop/something/openCV/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml"
#import cv2's params of face

save_path = "C:/Users/b1017089/Desktop/something/wrote/take_photo/photo/"

while(spsp):
    capture = cv2.VideoCapture(0)

    ret,image = capture.read()

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cascade = cv2.CascadeClassifier(cascade_path)

    facerect = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=2, minSize=(30, 30))

    color = (255,255,255)

    if(len(facerect) > 0):
        spsp = False
        for rect in facerect:
            cv2.rectangle(image, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), color, thickness=2)

        cv2.imwrite(save_path + "defolt_"+fileName+".png", image)
        src = cv2.imread(save_path + "defolt_"+fileName+".png", 1)
        image = src[facerect[0][1]-10:facerect[0][1]+facerect[0][3]+10, facerect[0][0]-10:facerect[0][0]+facerect[0][2]+10]

        cv2.imwrite(save_path + fileName + "'s face.png", image)
        print(fileName)
        age = face_ageer(save_path + fileName +"'s face.png") * 10
        print(age)
