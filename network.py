import tensorflow as tf
import hypar
import backbone_0 as nn

def Resnet_preprocess(x):
    return tf.keras.applications.resnet50.preprocess_input(x)

def Resnet():
    model = tf.keras.Sequential()
    model.add(tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape = hypar.input_shape))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.BatchNormalization(axis = -1, 
                                                 scale = True, 
                                                 momentum = 0.9, 
                                                 epsilon = 2e-5, 
                                                 gamma_regularizer=tf.keras.regularizers.l2(l=5e-4)))
    model.add(tf.keras.layers.Dense(512))
    model.trainable = True
    return model

def Resnet_nn():
    model = nn.ResNet(50)
    model.trainable = True
    return model