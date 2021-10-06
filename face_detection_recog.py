import tensorflow as tf
import numpy as np
import cv2
import os
import test_code

import arcface
import hypar
import network as net

class train_model(tf.keras.Model):
    def __init__(self):
        super(train_model, self).__init__()
        #self.resnet = net.Resnet_nn()
        self.resnet = net.Resnet()
        self.arcface = arcface.Arcface_Layer()

    def call(self, x, y):
        x = self.resnet(x)
        return self.arcface(x, y)

model = train_model()        
model.load_weights(file_name + '_full_model' + str(file_number)+ '.h5')

face_imgs_database = np.load(face_imgs_database.npy)
face_names_database = hypar.face_names_database
face_embedding_database = test_code.get_embeddings(face_imgs_database)

face_cascade = cv2.CascadeClassifier( os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml") )
cap = cv2.VideoCapture(1)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_color = frame[y:y + h, x:x + w]
        cv2.imwrite("filename.jpg", roi_color)
        names, faces = test_code.recognise(roi_color,face_imgs_database,face_embedding_database,face_names_database)
        print(names)


    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
