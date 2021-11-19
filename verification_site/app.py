from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
import numpy

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/model_resnet.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#model = tf.keras.Sequential()
#model.add(tf.keras.applications.ResNet50(weights="imagenet"))
#model.save(MODEL_PATH)
#print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    face1 = image.load_img(img_path, target_size=(224, 224))
    
    # Preprocessing the images
    x1 = image.img_to_array(face1)
    x1 = np.expand_dims(x1, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x1 = preprocess_input(x1, mode='caffe')

    preds = model.predict(x1)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/upl1', methods=['GET', 'POST'])
def upload1():
    if request.method == 'POST':
        fil = request.files['file']
        # Save the files to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads',"1")
        fil.save(file_path)
    return "DONE"

@app.route('/upl2', methods=['GET', 'POST'])
def upload2():
    if request.method == 'POST':

        fil = request.files['file2']
        # Save the files to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads',"2")
        fil.save(file_path)
    return "DONE"

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        basepath = os.path.dirname(__file__)

        # Make prediction
        preds = model_predict(os.path.join(basepath, 'uploads',"1"), model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        result = str(pred_class[0][0][1])               # Convert to string
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)


        