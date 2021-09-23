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
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# Instantiate an optimizer to train the model.
optimizer = tf.keras.optimizers.Adam(lr=hypar.learning_rate)

# Prepare the metrics.
train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

model = train_model()

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        logits = model(tf.slice(images, [0, 0, 0, 0], [hypar.batch_size, 112, 112, 3]), tf.slice(labels, [0], [hypar.batch_size]))
        for i in range(hypar.batch_multiplier - 1):
            logits = tf.concat([logits, model(tf.slice(images, [hypar.batch_size * (i + 1), 0, 0, 0], [hypar.batch_size, 112, 112, 3]), tf.slice(labels, [hypar.batch_size * (i + 1)], [hypar.batch_size]))], 0)
        pred = tf.nn.softmax(logits)
        inf_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        loss = inf_loss
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_acc_metric.update_state(labels, logits)
    train_loss = tf.reduce_mean(loss)
    accuracy = train_acc_metric.result()
    inference_loss = tf.reduce_mean(inf_loss)

    return accuracy, train_loss, inference_loss

images = load('x_train.npy')
labels = load('y_train.npy')

epochs = 2
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    start_time = time.time()

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        loss_value = train_step(x_batch_train, y_batch_train)

        # Log every 200 batches.
        if step % 200 == 0:
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, float(loss_value))
            )
            print("Seen so far: %d samples" % ((step + 1) * batch_size))

    print("Training acc over epoch: %.4f" % (float(train_acc),))

    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_states()

    print("Time taken: %.2fs" % (time.time() - start_time))