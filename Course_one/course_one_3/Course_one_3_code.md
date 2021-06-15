# cat classifier with 2-layer neural network
```python
"""
    cat, non-cat classification neural network with one hidden layer only written by numpy
    i will build a neural network like this:

                a1[1]
    x1
                a2[1]
       -------->       --------> a[2]    with sample of m
                a3[1]
    x2
                a4[1]
"""

import numpy as np
from Project.cat_classifier.lr_utils import load_dataset
from PIL import Image
import matplotlib.pyplot as plt


# flatten dataset
def flatten_data_set():
    x_train = train_set_x_orig.reshape((train_set_x_orig.shape[0], train_set_x_orig.shape[1] *
                                        train_set_x_orig.shape[2] * train_set_x_orig.shape[3])).T
    x_test = test_set_x_orig.reshape((test_set_x_orig.shape[0], test_set_x_orig.shape[1] *
                                      test_set_x_orig.shape[2] * test_set_x_orig.shape[3])).T
    return x_train/255, x_test/255


# initialize parameters
def init_parameters(X, n_hidden, Y):
    # get shape
    n_x = X.shape[0]
    n_y = Y.shape[0]

    # w1, w2, b1, b2
    w1 = np.random.randn(n_hidden, n_x) * 0.01
    b1 = np.zeros((n_hidden, 1))
    w2 = np.random.randn(n_y, n_hidden) * 0.01
    b2 = np.zeros((n_y, 1))

    # save into parameters
    parameters = {"W1": w1, "W2": w2, "b1": b1, "b2": b2}

    return parameters


# forward prop
def forward(X, parameters):

    Z1 = np.dot(parameters["W1"], X) + parameters["b1"]
    A1 = np.tanh(Z1)
    Z2 = np.dot(parameters["W2"], A1) + parameters["b2"]
    A2 = 1 / (1 + np.exp(-Z2))

    cache = {"Z1": Z1, "Z2:": Z2, "A1": A1, "A2": A2}
    return A2, cache


# backward prop
def backward(X, Y, cache, parameters):
    # get training sample
    m = X.shape[1]

    # backward prop
    dZ2 = cache["A2"] - Y
    dW2 = 1/m * np.dot(dZ2, cache["A1"].T)
    db2 = 1/m * np.sum(dZ2)
    dZ1 = np.dot(parameters['W2'].T, dZ2) * (1 - np.power(cache["A1"], 2))
    dW1 = 1/m * np.dot(dZ1, X.T)
    db1 = 1/m * np.sum(dZ1)

    grad = {"dW1": dW1, 'db1': db1, "dW2": dW2, "db2": db2}
    return grad


def update_parameters(grad, parameters, learning_rate):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    W1 -= learning_rate * grad["dW1"]
    b1 -= learning_rate * grad["db1"]
    W2 -= learning_rate * grad["dW2"]
    b2 -= learning_rate * grad["db2"]

    parameters = {"W1": W1, "W2": W2, "b1": b1, "b2": b2}
    return parameters


def predict(X, parameters):
    predictions = forward(X, parameters)[0]
    for i in range(0, len(predictions[0])):
        if predictions[0][i] >= 0.5:
            predictions[0][i] = 1
        else:
            predictions[0][i] = 0
    return predictions


# sum all of this
def nn_model(X, Y, n_hidden, learning_rate, iteration, show_cost=False):
    # init
    parameters = init_parameters(X, n_hidden, Y)
    # print('before', parameters)

    # iteration
    for iter in range(iteration):
        # forward
        A2, cache = forward(X, parameters)
        # print("A2", A2)
        # backward
        grad = backward(X, Y, cache, parameters)
        # update parameters
        parameters = update_parameters(grad, parameters, learning_rate)
        # new_cost
        cost = -1/X.shape[1] * np.sum(Y*np.log(A2) + (1-Y)*np.log(1-A2))
        if iter % 100 == 0 and show_cost:
            print('iter %d times, cost =' % iter, cost)

    return parameters


def image2prediction(my_image, parameters):
    fname = 'images/' + my_image
    im = Image.open(fname)
    im_resize = im.resize((64, 64))
    my_image = np.asarray(im_resize).reshape((1, 64*64*3)).T
    my_predicted_image = predict(my_image, parameters)

    plt.imshow(im)
    print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" +
          classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
    plt.show()


def test_accuracy(X, Y, parameters):
    prediction = predict(X, parameters)
    res = []
    for i in range(len(prediction[0])):
        if prediction[0][i] == Y[0][i]:
            res.append(1)
    print('accuracy : %2f' % ((len(res) / len(prediction[0])) * 100))


if __name__ == '__main__':
    # initialize cat_data
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
    X, X_test = flatten_data_set()
    # train
    parameters = nn_model(X, train_set_y, n_hidden=4, learning_rate=0.01, iteration=2000, show_cost=True)
    # predict
    test_accuracy(X_test, test_set_y, parameters)
    image2prediction('baidu_dog_1.jpeg', parameters)
```
