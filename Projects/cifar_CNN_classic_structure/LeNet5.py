"""
    LeNet
"""

import os
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, Model

# dataset
cifar = tf.keras.datasets.cifar10
(train_image, train_label), (test_image, test_label) = cifar.load_data()

train_image = train_image.reshape(train_image.shape[0], 32, 32, 3)
train_image, test_image = train_image / 255.0, test_image / 255.0


# LeNet5
class LeNet5(Model):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.c1 = layers.Conv2D(filters=6, kernel_size=(5, 5), activation='sigmoid')
        self.p1 = layers.MaxPooling2D(pool_size=(2, 2), strides=2)
        self.c2 = layers.Conv2D(filters=16, kernel_size=(5, 5), activation='sigmoid')
        self.p2 = layers.MaxPooling2D(pool_size=(2, 2), strides=2)
        self.flatten = layers.Flatten()
        self.d1 = layers.Dense(120, activation='sigmoid')
        self.d2 = layers.Dense(84, activation='sigmoid')
        self.out = layers.Dense(10, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.p1(x)
        x = self.c2(x)
        x = self.p2(x)

        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)

        y = self.out(x)
        return y


# create model
model = LeNet5()

# compile model
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

# save checkpoint
checkpoint_path = 'checkpoint/cifar10.ckpt'
# if os.path.exists(checkpoint_path + '.index'):
#     print('------------load model------------')
#     model.load_weights(checkpoint_path)

# define callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    save_best_only=True
)

# model fit
history = model.fit(train_image, train_label, epochs=10, validation_freq=1, validation_data=(test_image, test_label),
                    callbacks=[cp_callback])

# show summary
model.summary()

# write weights into txt
with open('train_data/LeNet5_weights.txt', 'w') as file:
    for v in model.trainable_variables:
        file.write(str(v.name) + '\n')
        file.write(str(v.shape) + '\n')
        file.write(str(v.numpy) + '\n')

# plot result
loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']

plt.subplot(1, 2, 1)
plt.title('Train and Validation Loss')
plt.xlabel('Epoch')
plt.plot(loss, label='Train loss')
plt.plot(val_loss, label='Validation loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.title('Train and Validation Accuracy')
plt.xlabel('Epoch')
plt.plot(acc, label='Train accuracy')
plt.plot(val_acc, label='Validation accuracy')
plt.legend()

plt.show()

