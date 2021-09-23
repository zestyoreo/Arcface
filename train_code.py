from arcface import Arcface_Layer
from network import Resnet
import tensorflow as tf
import hypar
import time
import numpy as np
from numpy import load

class train_model(tf.keras.Model):
    def __init__(self):
        super(train_model, self).__init__()
        self.resnet = Resnet()
        self.arcface = Arcface_Layer()

    def call(self, x, y):
        x = self.resnet(x)
        return self.arcface(x, y)

# Instantiate a loss function.
def loss_fxn(logits,labels):
    loss_fn = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    return loss_fn
    
# Instantiate an optimizer to train the model.
optimizer = tf.keras.optimizers.Adam(lr=hypar.learning_rate)

model = train_model()

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        logits = model(images,labels)
        pred = tf.nn.softmax(logits)
        inf_loss = loss_fxn(logits,labels)
        loss = inf_loss
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss = tf.reduce_mean(loss)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, axis=1, output_type=tf.dtypes.int32), labels), dtype=tf.float32))
    inference_loss = tf.reduce_mean(inf_loss)
    regularization_loss = 0
    return accuracy, train_loss, inference_loss, regularization_loss

images = load('x_train.npy')
labels = load('y_train.npy')
# Prepare the training dataset.
train_dataset = tf.data.Dataset.from_tensor_slices((images, labels))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(hypar.batch_size)

epochs = 2
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    start_time = time.time()

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        accuracy, train_loss, inference_loss, regularization_loss = train_step(x_batch_train, y_batch_train, hypar.reg_coef)
        # Log every 200 batches.
        if step % 200 == 0:
            print("Training loss (for one batch) at step %d: %.4f"% (step, float(train_loss)))
            print("Seen so far: %d samples" % ((step + 1) * hypar.batch_size))

    print("Training acc over epoch: %.4f" % (float(accuracy)))

    print("Time taken: %.2fs" % (time.time() - start_time))