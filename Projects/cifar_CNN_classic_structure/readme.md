# cifar10 datasets used diifferent classic CNN Model

```
# all structure
import tensorflow as tf
from keras import Model, Sequential
from keras.layers import *
```

# LeNet
1998

![image](https://user-images.githubusercontent.com/71109255/125166517-9983db00-e1ce-11eb-8148-68f6a96ae844.png)

```python
class LeNet5(Model):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.c1 = Conv2D(filters=6, kernel_size=(3, 3), activation='sigmoid')
        self.p1 = MaxPooling2D(pool_size=(2, 2), strides=2)
        
        self.c2 = Conv2D(filters=16, kernel_size=(3, 3), activation='sigmoid')
        self.p2 = MaxPooling2D(pool_size=(2, 2), strides=2)
        
        self.flatten = Flatten()
        self.f1 = Dense(units=120, activation='sigmoid')
        
        self.f2 = Dense(units=84, activation='sigmoid')
        
        self.out = Dense(units=10, activation='softmax')
        
    def call(self, x):
        x = self.c1(x)
        x = self.p1(x)
        
        x = self.c2(x)
        x = self.p2(x)
        
        x = self.flatten(x)
        x = self.f1(x)
        
        x = self.f2(x)
        
        out = self.f3(x)
        return out
```
# AlexNet
2012/9 ImageNet top-1 acc: 62.5%, parameters: 60M

![image](https://user-images.githubusercontent.com/71109255/125166466-4c076e00-e1ce-11eb-9ebd-7a8dd9531ffc.png)
```python
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
        out = self.out(x)

        return out
```

# VGG
2014/9 ImageNet top-1 acc: 74%, parameter: 144M

![image](https://user-images.githubusercontent.com/71109255/125168557-b32a2000-e1d8-11eb-8091-dbb152981253.png)

```python
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
```

# InceptionV1
2014/9 ImageNet top-1 acc: 69.8%, parameters: 5M

one single block

![image](https://user-images.githubusercontent.com/71109255/125167115-8292b800-e1d1-11eb-8b5e-bdb6eb75468e.png)

structure

![image](https://user-images.githubusercontent.com/71109255/125167767-b1f6f400-e1d4-11eb-989e-19fb1c76ea2a.png)

```python
class ConvBNRelu(Model):
    def __init__(self, filters, kernel_size=3, strides=1, padding='same'):
        super(ConvBNRelu, self).__init__()
        self.model = Sequential([
            Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding),
            BatchNormalization(),
            ReLU()
        ])
    
    def call(self, x):
        out = self.model(x)
        return out
        
        
class InceptionBlock(Model):
    def __init__(self, init_filters=16, strides=1, padding='same'):
        super(InceptionBlock, self).__init__()
        self.c1 = ConvBNRelu(filters=init_filters, kernel_size=1)
        
        self.c2_1 = ConvBNRelu(filters=init_filters, kernel_size=1)
        self.c2_2 = ConvBNRelu(filters=init_filters, kernel_size=3)
        
        self.c3_1 = ConvBNRelu(filters=init_filters, kernel_size=1)
        self.c3_2 = ConvBNRelu(filters=init_filters, kernel_size=5)
        
        self.p4_1 = MaxPooling2D((3, 3), padding='same')
        self.c4_2 = ConvBNRelu(filters=init_filters, kernel_size=1)
        
    def call(self, x):
        x1 = self.c1(x)
        
        x2 = self.c2_1(x)
        x2 = self.c2_2(x2)
        
        x3 = self.c3_1(x)
        x3 = self.c3_2(x3)
        
        x4 = self.p4_1(x)
        x4 = self.c4_2(x4)
        
        out = tf.concat([x1, x2, x3, x4], axis=3)
        return out


# use 4 blocks, first two blocks use 16 filters and last two use 32 filters, one and three InceptionBlocks use strides=2
class Inception10(Model):
    def __init__(self, init_filters=16, num_blocks=2):
        super(Inception10, self).__init__()
        self.out_filters = init_filters
        
        self.c1 = ConvBNRelu(filters=init_filters)
        
        self.blocks = Sequential()
        for block_id in range(num_blocks):
            for layer_id in range(2):
                if layer_id == 0:
                    block = InceptionBlock(filters=self.out_filters, strides=2)
                else:
                    block = InceptionBlock(filters=self.out_filters, strides=1)
                self.blocks.add(block)
            self.out_filters *= 2
        
        self.p1 = GlobalAveragePooling2D()
        self.out = Dense(10, activation='softmax')
    
    def call(self, x):
        x = self.c1(x)
        x = self.blocks(x)
        x = self.p1(x)
        out = self.out(x)
        return out
```
# ResNet152
2015/12 ImageNet top-1 acc: 78.6%

![image](https://user-images.githubusercontent.com/71109255/125168028-fafb7800-e1d5-11eb-9c40-c2175886349d.png)

structure

![image](https://user-images.githubusercontent.com/71109255/125167992-de5f4000-e1d5-11eb-920e-c8e5d7d363f7.png)

```python
# two different types of path, one is same strides=1, one is strides=2 and strides=1, need conv(1, 1) filter to fix it
class ResNetBlock(Model):
    def __init__(self, filters, kernel_size=3, strides=1, residual_path=False)
        super(ResNetBlock, self).__init__()
        self.residual_path = residual_path
        
        self.c1 = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')
        self.b1 = BatchNormalization()
        self.a1 = ReLU()
        
        self.c2 = Conv2D(filters=filters, kernel_size=kernel_size, strides=1, padding='same')
        self.b2 = BatchNormalization()
        
        if self.residual_path:
            self.c1_down = Conv2D(filters=filters, kernel_size=1, strides=strides, padding='same')
            self.b1_down = BatchNormalization()
            
        self.a2 = ReLU()
        
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


class ResNet18(Model):
    def __init__(self, block_list, initial_filters=64):
        super(ResNet18, self).__init__()
        self.out_filters = initial_filters

        self.c1 = Conv2D(initial_filters, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal')
        self.b1 = BatchNormalization()
        self.a1 = ReLU()

        self.blocks = Sequential()
        for block_id in range(len(block_list)):
            for layer_id in range(block_list[block_id]):
                if block_id != 0 and layer_id == 0:
                    block = ResNetBlock(self.out_filters, strides=2, residual_path=True)
                else:
                    block = ResNetBlock(self.out_filters, strides=1, residual_path=False)
                self.blocks.add(block)
            self.out_filters *= 2

        self.p1 = GlobalAvgPool2D()
        self.f1 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)

        x = self.blocks(x)
        x = self.p1(x)
        out = self.f1(x)

        return out
```
