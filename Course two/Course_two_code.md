# cat classifier with logistic regreesion
```python
"""
    cat-classifier with logistics regression: (just one single neural)
    1. loading dataset and flatten them.
    2. initialize w and b
    3. forward prop: calculate Z and A (y_hat)
    4. backward prop: calculate dw, db
    5. update w and b
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from Project.cat_classifier.lr_utils import load_dataset
from matplotlib.pyplot import imread

# loading dataset
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()


# flatten dataset
def flatten_data_set():
    x_train = train_set_x_orig.reshape((train_set_x_orig.shape[0], train_set_x_orig.shape[1] *
                                        train_set_x_orig.shape[2] * train_set_x_orig.shape[3])).T
    x_test = test_set_x_orig.reshape((test_set_x_orig.shape[0], test_set_x_orig.shape[1] *
                                      test_set_x_orig.shape[2] * test_set_x_orig.shape[3])).T
    return x_train/255, x_test/255


# activation function sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# init w, b
def init_w_b():
    w = np.zeros((train_set_x_orig.shape[1] * train_set_x_orig.shape[2] * train_set_x_orig.shape[3], 1))
    b = 0
    return w, b


# propagate
def propagate(w, b, X, Y):
    # init
    m = X.shape[1]

    # forward prop
    Z = np.dot(w.T, X) + b
    A = sigmoid(Z)
    # cost = -1/m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    # backward prop:
    dZ = A - Y
    dw = 1/m * np.dot(X, dZ.T)
    db = 1/m * np.sum(dZ)

    # record grads
    grads = {"dw": dw, "db": db}

    return grads


# optimize
def optimize(w, b, X, Y, learning_rate):
    grads = propagate(w, b, X, Y)
    w = w - learning_rate * grads["dw"]
    b = b - learning_rate * grads["db"]

    new_grads = {'w': w, 'b': b}
    return new_grads


# predict
def predict(w, b, X):
    A = sigmoid(np.dot(w.T, X) + b)
    Y_preidct = np.zeros((A.shape))

    for i in range(A.shape[1]):
        if A[0][i] >= 0.5:
            Y_preidct[0][i] = 1
        else:
            Y_preidct[0][i] = 0
    return Y_preidct


# model
def model(w, b, X, X_test, Y, Y_test, learning_rate, iter_num):
    costs = []
    for iter in range(iter_num):

        grads = optimize(w, b, X, Y, learning_rate)
        w = grads['w']
        b = grads['b']
        cost = -1/X.shape[1] * np.sum(Y * np.log(sigmoid(np.dot(w.T, X) + b)) + (1 - Y) *
                                      np.log(1 - sigmoid(np.dot(w.T, X) + b)))

        if iter % 50 == 0:
            print('training {} times cost:'.format(iter), cost)
            costs.append(cost)

    # predict y
    y_hat = predict(w, b, X)
    y_pred_hat = predict(w, b, X_test)
    print('train accuracy: {}%'.format(100 - np.mean(np.abs(y_hat - Y)) * 100))
    print('test accuracy: {}%'.format(100 - np.mean(np.abs(y_pred_hat - Y_test)) * 100))
    costs = np.squeeze(costs)
    plt.plot(costs)
    plt.show()
    return w, b


# my_image2predict
def image2prediction(my_image, w, b):
    fname = 'images/' + my_image
    im = Image.open(fname)
    im_resize = im.resize((64, 64))
    my_image = np.asarray(im_resize).reshape((1, 64*64*3)).T
    print(my_image.shape)
    my_predicted_image = predict(w, b, my_image)

    plt.imshow(im)
    print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
    plt.show()


if __name__ == '__main__':
    w, b = init_w_b()
    train_set_x_orig, test_set_x_orig = flatten_data_set()
    print('train_set_x_orig:', train_set_x_orig.shape)
    print('train_set_y:', train_set_y.shape)
    w, b = model(w, b, train_set_x_orig, test_set_x_orig, train_set_y, test_set_y, learning_rate=0.01, iter_num=3000)
    image2prediction('dg_cat.jpg', w, b)

```
