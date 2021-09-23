import tensorflow as tf
import hypar

def Resnet():
    model = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape = hypar.input_shape)
    model.trainable = True
    return model