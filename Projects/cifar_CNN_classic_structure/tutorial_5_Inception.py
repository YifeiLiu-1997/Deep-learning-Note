"""
    Inception
"""

import os

import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, Model, Sequential
from keras.layers import *

# dataset
cifar = tf.keras.datasets.cifar10
(train_image, train_label), (test_image, test_label) = cifar.load_data()

train_image = train_image.reshape(train_image.shape[0], 32, 32, 3)
train_image, test_image = train_image / 255.0, test_image / 255.0


# Inception Quick Structure
class ConvBNRelu(Model):
    def __init__(self, ch, kernel_size=(3, 3), strides=1, padding='same'):
        super(ConvBNRelu, self).__init__()
        self.model = Sequential([
            layers.Conv2D(filters=ch, kernel_size=kernel_size, strides=strides, padding=padding),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

    def call(self, x):
        x = self.model(x)
        return x


# Inception Block
class InceptionBlock(Model):
    def __init__(self, ch, strides=1):
        super(InceptionBlock, self).__init__()
        # 1*1 block
        self.c1 = ConvBNRelu(ch, kernel_size=(1, 1))
        # 1*1 3*3 block
        self.c2_1 = ConvBNRelu(ch, kernel_size=(1, 1))
        self.c2_2 = ConvBNRelu(ch, kernel_size=(3, 3))
        # 1*1 5*5 block
        self.c3_1 = ConvBNRelu(ch, kernel_size=(1, 1))
        self.c3_2 = ConvBNRelu(ch, kernel_size=(5, 5))
        # 3*3 maxpool, 1*1 block
        self.p4_1 = layers.MaxPooling2D((3, 3), strides=1, padding='same')
        self.c4_2 = ConvBNRelu(ch, kernel_size=(1, 1))

    def call(self, x):
        x1 = self.c1(x)
        x2 = self.c2_1(x)
        x2 = self.c2_2(x2)
        x3 = self.c3_1(x)
        x3 = self.c3_2(x3)
        x4 = self.p4_1(x)
        x4 = self.c4_2(x4)
        # output -> concat all block
        y = tf.concat([x1, x2, x3, x4], axis=3)
        return y


# Inception10
class Inception10(Model):
    def __init__(self, num_blocks, num_classes, init_ch=16, **kwargs):
        super(Inception10, self).__init__()
        self.c1 = ConvBNRelu(init_ch)
        self.blocks = Sequential()
        for block_id in range(num_blocks):
            for layers_id in range(2):
                if layers_id == 0:
                    block = InceptionBlock(ch=init_ch, strides=2)
                else:
                    block = InceptionBlock(ch=init_ch, strides=1)
                self.blocks.add(block)
            init_ch *= 2
        self.p1 = layers.GlobalAvgPool2D()
        self.f1 = layers.Dense(10, activation='softmax')

    def call(self, x):
        x = self.c1(x)         # 32, 32, 3 -> 32, 32, 16
        x = self.blocks(x)     # 32, 32, 16 -> 8, 8, 128
        x = self.p1(x)         # 8, 8, 128 -> 1, 1, 128
        y = self.f1(x)         # fc
        return y


# create model
model = Inception10(num_blocks=2, num_classes=10)

# compile model0
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
with open('train_data/Inception10_weights.txt', 'w') as file:
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
