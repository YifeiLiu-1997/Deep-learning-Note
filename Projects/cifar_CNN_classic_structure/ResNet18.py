"""
    ResNet
"""

import os
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, Model, Sequential

# dataset
cifar = tf.keras.datasets.cifar10
(train_image, train_label), (test_image, test_label) = cifar.load_data()

train_image = train_image.reshape(train_image.shape[0], 32, 32, 3)
train_image, test_image = train_image / 255.0, test_image / 255.0


# ResNet Block
class ResNetBlock(Model):
    def __init__(self, filters, strides=1, residual_path=False):
        super(ResNetBlock, self).__init__()
        self.residual_path = residual_path

        self.c1 = layers.Conv2D(filters, (3, 3), strides=strides, padding='same')
        self.b1 = layers.BatchNormalization()
        self.a1 = layers.ReLU()

        self.c2 = layers.Conv2D(filters, (3, 3), strides=1, padding='same')
        self.b2 = layers.BatchNormalization()

        if residual_path:
            self.c1_down = layers.Conv2D(filters, (1, 1), strides=strides, padding='same')
            self.b1_down = layers.BatchNormalization()

        self.a2 = layers.ReLU()

    def call(self, x):
        residual = x

        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)

        x = self.c2(x)
        x = self.b2(x)

        if self.residual_path:
            residual = self.c1_down(residual)
            residual = self.b1_down(residual)

        out = self.a2(residual + x)
        return out


# ResNet
class ResNet18(Model):
    def __init__(self, block_list, initial_filters=64):
        super(ResNet18, self).__init__()
        self.out_filters = initial_filters

        self.c1 = layers.Conv2D(initial_filters, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal')
        self.b1 = layers.BatchNormalization()
        self.a1 = layers.ReLU()

        self.blocks = Sequential()
        for block_id in range(len(block_list)):
            for layer_id in range(block_list[block_id]):
                if block_id != 0 and layer_id == 0:
                    block = ResNetBlock(self.out_filters, strides=2, residual_path=True)
                else:
                    block = ResNetBlock(self.out_filters, strides=1, residual_path=False)
                self.blocks.add(block)
            self.out_filters *= 2

        self.p1 = layers.GlobalAvgPool2D()
        self.f1 = layers.Dense(10, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)

        x = self.blocks(x)
        x = self.p1(x)
        y = self.f1(x)

        return y


# create model
model = ResNet18(block_list=[2, 2, 2, 2])

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
with open('train_data/ResNet18_weights.txt', 'w') as file:
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
