import tensorflow as tf
import math
import time
import numpy as np

import hypar
import backbone_0 as nn
import network as net

X = np.load('datasets/x_train.npy', allow_pickle=True)
Y = np.load('datasets/y_train.npy', allow_pickle=True)

X = np.array(X, dtype='float32')
Y = np.array(Y, dtype='int32')
Y = np.reshape(Y, Y.shape[0])
print("X shape:",X.shape,"Y shape:",Y.shape) 
X = net.Resnet_preprocess(X)
images = X
labels = Y
hypar.batch_size = 64
# Prepare the training dataset.
train_dataset = tf.data.Dataset.from_tensor_slices((images, labels))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(hypar.batch_size)
print("Training dataset ready!")

num_classes = hypar.num_classes #number of people in the dataset

class Arcface_Layer(tf.keras.layers.Layer):                                     # Arcface layer definition
    def __init__(self, num_outputs = num_classes, s=64., m=0.5):                 # s is scale factor, m is the margin to be added to the angle 'theta'
        self.output_dim = num_outputs
        self.s = s
        self.m = m
        super(Arcface_Layer, self).__init__()

    def build(self, input_shape):

        self.kernel = self.add_weight(name='weight',
                                          shape=(input_shape[-1],self.output_dim),
                                          initializer='glorot_uniform',
                                          regularizer=tf.keras.regularizers.l2(l=5e-4),
                                          trainable=True)
        super(Arcface_Layer, self).build(input_shape)


    def call(self, embedding, labels):
        cos_m = math.cos(self.m)
        sin_m = math.sin(self.m)
        mm = sin_m * self.m  # issue 1
        threshold = math.cos(math.pi - self.m)
        # inputs and weights norm
        embedding_norm = tf.norm(embedding, axis=1, keepdims=True)
        embedding = embedding / embedding_norm
        weights_norm = tf.norm(self.kernel, axis=0, keepdims=True)
        weights = self.kernel / weights_norm
        # cos(theta+m)
        cos_t = tf.matmul(embedding, weights, name='cos_t')
        cos_t2 = tf.square(cos_t, name='cos_2')
        sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
        sin_t = tf.sqrt(sin_t2, name='sin_t')
        cos_mt = self.s * tf.subtract(tf.multiply(cos_t, cos_m),
                                      tf.multiply(sin_t, sin_m), name='cos_mt')

        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_t - threshold
        cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)

        keep_val = self.s * (cos_t - mm)
        cos_mt_temp = tf.where(cond, cos_mt, keep_val)

        mask = tf.one_hot(labels, depth=self.output_dim, name='one_hot_mask')
        # mask = tf.squeeze(mask, 1)
        inv_mask = tf.subtract(1., mask)

        s_cos_t = tf.multiply(self.s, cos_t, name='scalar_cos_t')

        output = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, mask), name='arcface_loss_output')

        return output

class train_model(tf.keras.Model):
    def __init__(self):
        super(train_model, self).__init__()
        #self.resnet = net.Resnet_nn()
        self.resnet = net.Resnet()
        self.arcface = Arcface_Layer()

    def call(self, x, y):
        x = self.resnet(x)
        return self.arcface(x, y)

# Instantiate a loss function.
def loss_fxn(logits,labels):
    loss_fn = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    return loss_fn
    
# Instantiate an optimizer to train the model.
learning_rate = 0.0005
#optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=False)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,name='Adam')

model = train_model()

@tf.function
def train_step(images, labels, regCoef):
    with tf.GradientTape() as tape:
        logits = model(images,labels)
        pred = tf.nn.softmax(logits)
        #inf_loss = loss_fxn(pred,labels)
        inf_loss = loss_fxn(logits,labels)
        reg_loss = tf.add_n(model.losses)
        loss = inf_loss + reg_loss * regCoef
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss = tf.reduce_mean(loss)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, axis=1, output_type=tf.dtypes.int32), tf.cast(labels,dtype = tf.int32)), dtype=tf.float32))
    inference_loss = tf.reduce_mean(inf_loss)
    regularization_loss = tf.reduce_mean(reg_loss)
    return accuracy, train_loss, inference_loss, regularization_loss

loss_log = []

epochs = 50
reg_coef = 1.0
file_number = 1

for save_wt in range (0, 2):
  for epoch in range(epochs):
      print("\nStart of epoch %d" % (epoch,))
      start_time = time.time()

      # Iterate over the batches of the dataset.
      for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
          accuracy, train_loss, inference_loss, regularization_loss = train_step(x_batch_train, y_batch_train,reg_coef)
          if step % 20 == 0:
            loss_log.append(train_loss)
            template = 'Epoch {}, Step {}, Loss: {}, Reg loss: {}, Accuracy: {}'
            print(template.format(epoch + 1, step,
                                  '%.5f' % (inference_loss),
                                  '%.5f' % (regularization_loss),
                                  '%.5f' % (accuracy)))
      
  file_number += 1
  file_name = 'model_weights_'
  file_name = file_name + str(file_number*epochs)+ '_epochs'
  model.save(file_name)