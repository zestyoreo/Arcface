from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
import numpy
from tensorflow.python.keras.engine import training

# Keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/arcface_weights.h5'
cosine_threshold = 0.08

def ResNet34():

	img_input = tf.keras.layers.Input(shape=(112, 112, 3))

	x = tf.keras.layers.ZeroPadding2D(padding=1, name='conv1_pad')(img_input)
	x = tf.keras.layers.Conv2D(64, 3, strides=1, use_bias=False, kernel_initializer='glorot_normal', name='conv1_conv')(x)
	x = tf.keras.layers.BatchNormalization(axis=3, epsilon=2e-5, momentum=0.9, name='conv1_bn')(x)
	x = tf.keras.layers.PReLU(shared_axes=[1, 2], name='conv1_prelu')(x)
	x = stack_fn(x)

	model = training.Model(img_input, x, name='ResNet34')

	return model

def block1(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
	bn_axis = 3

	if conv_shortcut:
		shortcut = tf.keras.layers.Conv2D(filters, 1, strides=stride, use_bias=False, kernel_initializer='glorot_normal', name=name + '_0_conv')(x)
		shortcut = tf.keras.layers.BatchNormalization(axis=bn_axis, epsilon=2e-5, momentum=0.9, name=name + '_0_bn')(shortcut)
	else:
		shortcut = x

	x = tf.keras.layers.BatchNormalization(axis=bn_axis, epsilon=2e-5, momentum=0.9, name=name + '_1_bn')(x)
	x = tf.keras.layers.ZeroPadding2D(padding=1, name=name + '_1_pad')(x)
	x = tf.keras.layers.Conv2D(filters, 3, strides=1, kernel_initializer='glorot_normal', use_bias=False, name=name + '_1_conv')(x)
	x = tf.keras.layers.BatchNormalization(axis=bn_axis, epsilon=2e-5, momentum=0.9, name=name + '_2_bn')(x)
	x = tf.keras.layers.PReLU(shared_axes=[1, 2], name=name + '_1_prelu')(x)

	x = tf.keras.layers.ZeroPadding2D(padding=1, name=name + '_2_pad')(x)
	x = tf.keras.layers.Conv2D(filters, kernel_size, strides=stride, kernel_initializer='glorot_normal', use_bias=False, name=name + '_2_conv')(x)
	x = tf.keras.layers.BatchNormalization(axis=bn_axis, epsilon=2e-5, momentum=0.9, name=name + '_3_bn')(x)

	x = tf.keras.layers.Add(name=name + '_add')([shortcut, x])
	return x

def stack1(x, filters, blocks, stride1=2, name=None):
	x = block1(x, filters, stride=stride1, name=name + '_block1')
	for i in range(2, blocks + 1):
		x = block1(x, filters, conv_shortcut=False, name=name + '_block' + str(i))
	return x

def stack_fn(x):
	x = stack1(x, 64, 3, name='conv2')
	x = stack1(x, 128, 4, name='conv3')
	x = stack1(x, 256, 6, name='conv4')
	return stack1(x, 512, 3, name='conv5')

def loadModel():
	base_model = ResNet34()
	inputs = base_model.inputs[0]
	arcface_model = base_model.outputs[0]
	arcface_model = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=2e-5)(arcface_model)
	arcface_model = tf.keras.layers.Dropout(0.4)(arcface_model)
	arcface_model = tf.keras.layers.Flatten()(arcface_model)
	arcface_model = tf.keras.layers.Dense(512, activation=None, use_bias=True, kernel_initializer="glorot_normal")(arcface_model)
	embedding = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=2e-5, name="embedding", scale=True)(arcface_model)
	model = tf.keras.models.Model(inputs, embedding, name=base_model.name)
	
	model.load_weights(MODEL_PATH)

	return model

# Load your trained model
model = loadModel()
print("ArcFace expects ",model.layers[0].input_shape[1:]," inputs")
print("and it represents faces as ", model.layers[-1].output_shape[1:]," dimensional vectors")


def get_distance(emb1,emb2):
  """
  emb1 & emb2: are both 512 dimensional vectors from the trained resnet model

  get_distance: returns cosine_distance
  Check Out "https://github.com/zestyoreo/Arcface/blob/main/get_distance()_test.ipynb" for clarity
  """

  a = np.matmul(np.transpose(emb1), emb2)
  b = np.sum(np.multiply(emb1, emb1))
  c = np.sum(np.multiply(emb2, emb2))
  cosine_distance = 1 - (a / (np.sqrt(b) * np.sqrt(c)))

  return cosine_distance

def preprocess_face(img, target_size=(224, 224)):

	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# img = cv2.resize(img, target_size) #resize causes transformation on base image, adding black pixels to resize will not deform the base image
	
	if img.shape[0] > 0 and img.shape[1] > 0:
		factor_0 = target_size[0] / img.shape[0]
		factor_1 = target_size[1] / img.shape[1]
		factor = min(factor_0, factor_1)
		
		dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
		img = cv2.resize(img, dsize)
		
		# Then pad the other side to the target size by adding black pixels
		diff_0 = target_size[0] - img.shape[0]
		diff_1 = target_size[1] - img.shape[1]
		
		img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2)), 'constant')
   
	if img.shape[0:2] != target_size:
		img = cv2.resize(img, target_size)

	#normalizing the image pixels

	img_pixels = image.img_to_array(img) #what this line doing? must?
	img_pixels = np.expand_dims(img_pixels, axis = 0)
	img_pixels /= 255 #normalize input in [0, 1]

	return img_pixels


def face_verify(img_path,img_path2,model):
    face1 = image.load_img(img_path, target_size=(224, 224))
    face2 = image.load_img(img_path2, target_size=(224, 224))
    
    # Preprocessing the images
    x1 = image.img_to_array(face1)
    x2 = image.img_to_array(face2)

    x1 = np.expand_dims(x1, axis=0)
    x2 = np.expand_dims(x2, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x1 = preprocess_face(x1, mode='caffe')
    x2 = preprocess_face(x2, mode='caffe')

    embedding1 = model.predict(x1)
    embedding2 = model.predict(x2)
    preds = "Different People"

    cosine_distance = get_distance(embedding1,embedding2)
    if cosine_distance<cosine_threshold:
        preds = "Same People"

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

        # Make prediction (preds is a string telling same face or diff face)
        preds = face_verify(os.path.join(basepath, 'uploads',"1"),os.path.join(basepath, 'uploads',"2"), model)

        return preds
    return None


if __name__ == '__main__':
    app.run(debug=True)


        