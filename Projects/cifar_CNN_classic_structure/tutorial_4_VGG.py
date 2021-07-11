"""
    VGG16
"""

import os
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import Model, Sequential
from keras.layers import *

# dataset
cifar = tf.keras.datasets.cifar10
(train_image, train_label), (test_image, test_label) = cifar.load_data()

train_image = train_image.reshape(train_image.shape[0], 32, 32, 3)
train_image, test_image = train_image / 255.0, test_image / 255.0


# VGG16
class ConvBNRelu(Model):
    def __init__(self, filters=64, padding='same', strides=1, pooling=False, dropout=False):
        super(ConvBNRelu, self).__init__()
        self.model = Sequential([
            Conv2D(filters=filters, kernel_size=(3, 3), padding=padding, strides=strides),
            BatchNormalization(),
            ReLU()
        ])

        if pooling and dropout:
            self.model.add(MaxPooling2D((2, 2), strides=2))
            self.model.add(Dropout(0.2))

    def call(self, x):
        out = self.model(x)
        return out


class VGG16(Model):
    def __init__(self, block_list, init_filters=64):
        super(VGG16, self).__init__()
        self.out_filters = init_filters

        self.blocks = Sequential()
        for layers in block_list:
            for layers_id in range(layers):
                if layers_id == layers - 1:
                    block = ConvBNRelu(filters=self.out_filters, pooling=True, dropout=True)
                else:
                    block = ConvBNRelu(filters=self.out_filters)
                self.blocks.add(block)
            self.out_filters *= 2

        for layers in range(3):
            if layers == 2:
                block = ConvBNRelu(filters=512, pooling=True, dropout=True)
            else:
                block = ConvBNRelu(filters=512)
            self.blocks.add(block)

        self.flatten = Flatten()

        self.f1 = Dense(512, activation='relu')
        self.d1 = Dropout(0.2)

        self.f2 = Dense(512, activation='relu')
        self.d2 = Dropout(0.2)

        self.out = Dense(10, activation='softmax')

    def call(self, x):
        x = self.blocks(x)
        x = self.flatten(x)
        x = self.f1(x)
        x = self.d1(x)
        x = self.f2(x)
        x = self.d2(x)
        out = self.out(x)
        return out


# create model
model = VGG16(block_list=[2, 2, 3, 3])

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
history = model.fit(train_image, train_label, epochs=1, validation_freq=1, validation_data=(test_image, test_label),
                    callbacks=[cp_callback])

# show summary
model.summary()

# write weights into txt
with open('train_data/VGG16_weights.txt', 'w') as file:
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
