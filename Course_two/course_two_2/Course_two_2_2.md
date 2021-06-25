# Tuning process
- finding a good set up
- if you have two hyperparameters to tune, you can make a graph that x-axis is Hparameter1, y-axis is HP2, randomly choose for example 25 points to try which is the best.


# Using an appropriate scale to pick hyperparameters
-  when choosing learning_rate, if you random pass through 0.0001 - 1, so 90% num will in 0.1 - 1, but you just want 0.0001 - 0.1, so you can use 10^-n, to initialize your scale

# Batch Normalization
![image](https://user-images.githubusercontent.com/71109255/123383224-81546f00-d5c5-11eb-9497-2d5ba8d95be3.png)
- why does it work
- "covariate shift": if x mapping y, and when x's distribution change, you need retrain.
- because it can force each layer's Z[l] means and variance be a constant.
- because it can reduce "covariate shift" and connection with each layer.
- because it can make each layer learn by themselves more indenpendently.
![image](https://user-images.githubusercontent.com/71109255/123389091-19555700-d5cc-11eb-9d95-c1919a33d56f.png)

## Test time
- when test, using exponentially weight average(across mini-batch) to compute your Z_norm of X_test
![image](https://user-images.githubusercontent.com/71109255/123389953-fd9e8080-d5cc-11eb-8331-40e5ae99c156.png)

# softmax
![image](https://user-images.githubusercontent.com/71109255/123391144-4b67b880-d5ce-11eb-85bb-dd051e7e28af.png)

# Train softmax classifier
![image](https://user-images.githubusercontent.com/71109255/123393370-9aaee880-d5d0-11eb-86e2-84a1bb369d82.png)

# framework
- Caffe/Caffe2
- CNTK
- DL4L
- Keras
- Lasagne
- mxnet
- PaddlePaddle
- TensorFlow
- Theano
- Torch
## choosing tips
- Speed
- easy to code
- user's number

# TensorFlow
