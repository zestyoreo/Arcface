import numpy as np
import glob
from numpy import asarray
from numpy import save
import cv2
import os

#names = ["Amit", "atharva", "Balasubramanian", "Harshil", "Meghna", "Rihan Aaron", "Ronit", "Saad", "shubham"]
names = ["Ronit"]
down_size = 112
down_points = (down_size, down_size)

face_cascade = cv2.CascadeClassifier( os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml") )
face_database = {}
face_names_database = []

for name in names:
    file_name = "UMIC Members/"+name+"/*.jpeg"
    face_shots = []
    for file in glob.glob(file_name):
        photo = cv2.imread(file)
        gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            frame = cv2.rectangle(photo, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face_img = frame[y:y + h, x:x + w]
            face_img = cv2.resize(face_img, down_points, interpolation= cv2.INTER_LINEAR)
            face_img = asarray(face_img)
            print(face_img.shape)



cv2.destroyAllWindows()