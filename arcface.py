import tensorflow as tf
import math

num_classes = 13000 #number of people in the dataset
initializer = ""

#feature vector dimension = (512) [comes from the resnet model]


class Arcface_Layer(tf.keras.layers.Layer): #Arcface layer definition
  def __init__(self, num_outputs = num_classes, s=64, m=0.5):   #s is scale factor, m is the margin to be added to the angle 'theta'
    super(Arcface_Layer, self).__init__()
    self.num_outputs = num_outputs
    self.s = s
    self.m = m

  def build(self, input_shape):
    
    self.kernel = self.add_weight(name='weight',
                                      shape=(input_shape[-1],self.num_outputs),
                                      initializer='glorot_uniform',
                                      regularizer=tf.keras.regularizers.l2(l=5e-4),
                                      trainable=True)
    super(Arcface_Layer, self).build(input_shape)


  def call(self, feature_vec, ground_truth_vec):   #inputs is the 512 dimension feature vector
    gt = ground_truth_vec  #dims of gt is (num_classes)
    # feature vector and weights norm
    x = feature_vec
    norm_x = tf.norm(feature_vec, axis=1, keepdims=True)
    norm_W = tf.norm(self.kernel, axis=0, keepdims=True)

    x = x/norm_x
    W = self.kernel/norm_W

    cos_theta = tf.matmul(x,W)
    sin_theta = tf.math.sqrt(1-tf.math.square(cos_theta))
    theta = 

    cos_m = tf.math.cos(self.m)
    sin_m = 



    
    
    return 
