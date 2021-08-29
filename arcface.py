import tensorflow as tf
import math

num_classes = 13000 #number of people in the dataset
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
    
    mask = ground_truth_vec  #dims of mask is (num_classes), it is a one-hot vector
    inv_mask = tf.subtract(1., mask)

    # feature vector and weights norm
    x = feature_vec
    norm_x = tf.norm(feature_vec, axis=1, keepdims=True)
    norm_W = tf.norm(self.kernel, axis=0, keepdims=True)
    x = tf.math.divide(x, norm_x)
    W = tf.math.divide(self.kernel/norm_W)
    
    cos_theta = tf.matmul(tf.transpose(W),x)      # logit of  W.t*x 
    theta = tf.math.acos(cos_theta)   # all angle between each class' weight and x
    theta_class = tf.multiply(theta,mask)            #increasing angle theta of the class x belongs to alone
    theta_class_added_margin = theta_class + self.m
    theta_class_added_margin = theta_class_added_margin*mask
    cos_theta_margin = tf.math.cos(theta_class_added_margin)
    s_cos_t = tf.multiply(self.s, cos_theta_margin)
    s_cos_j = tf.multiply(self.s,tf.multiply(cos_theta,inv_mask))

    output = tf.add(s_cos_t,s_cos_j)

    return output
