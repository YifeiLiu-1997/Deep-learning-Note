"""
    AlexNet
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


# AlexNet8
class AlexNet(Model):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.c1 = layers.Conv2D(96, (3, 3))
        self.b1 = layers.BatchNormalization()
        self.a1 = layers.ReLU()
        self.p1 = layers.MaxPooling2D((3, 3), 2)
        self.c2 = layers.Conv2D(256, (3, 3))
        self.b2 = layers.BatchNormalization()
        self.a2 = layers.ReLU()
        self.p2 = layers.MaxPooling2D((3, 3), 2)
        self.c3 = layers.Conv2D(384, (3, 3), padding='same', activation='relu')
        self.c4 = layers.Conv2D(384, (3, 3), padding='same', activation='relu')
        self.c5 = layers.Conv2D(384, (3, 3), padding='same', activation='relu')
        self.p5 = layers.MaxPooling2D((3, 3), 2)
        self.flatten = layers.Flatten()
        self.d1 = layers.Dense(2048, activation='relu')
        self.dropout1 = layers.Dropout(0.5)
        self.d2 = layers.Dense(2048, activation='relu')
        self.dropout2 = layers.Dropout(0.5)
        self.out = layers.Dense(10, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)
        x = self.c2(x)
        x = self.b2(x)
        x = self.a2(x)
        x = self.p2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        x = self.p5(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.dropout1(x)
        x = self.d2(x)
        x = self.dropout2(x)
        y = self.out(x)

        return y


# create model
model = AlexNet()

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
with open('train_data/AlexNet_weights.txt', 'w') as file:
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

