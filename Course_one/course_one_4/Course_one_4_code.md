```python
"""
    deep learning first project: L-layers-nn-model to classify cat
    datasets: train_catvnoncat.h5
    testsets: test_catvnoncat.h5
    model: full-connected-nn-l-layers-neural-network
    only written with numpy
"""

from Project.cat_classifier.dnn_app_utils_v2 import *
import matplotlib.pyplot as plt
from PIL import Image
import pickle

# load datasets
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

# reshape the training and test examples
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# standardize data to have feature values between 0 and 1
train_x = train_x_flatten / 255
test_x = test_x_flatten / 255


# main function of L_layer_nn_model
def initialize_parameters(layers_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1])
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters


def linear_forward(A, W, b):
    Z = np.dot(W, A) + b

    cache = (A, W, b)

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):

    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)],
                                             parameters["b" + str(l)], activation='relu')
        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters["W" + str(L)],
                                          parameters["b" + str(L)], activation='sigmoid')
    caches.append(cache)

    return AL, caches


def compute_cost(AL, Y):
    m = Y.shape[1]

    cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    cost = np.squeeze(cost)

    return cost


def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1 / m * np.dot(dZ, A_prev.T)
    db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache

    if activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    grads["dA" + str(L)] = dAL
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward\
        (dAL, caches[-1], activation='sigmoid')

    for l in reversed(range(L - 1)):

        grads['dA' + str(l)], grads["dW" + str(l + 1)], grads["db" + str(l + 1)] = linear_activation_backward\
            (grads['dA' + str(l + 1)], caches[l], activation='relu')

    return grads


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2

    for l in range(L):
        parameters['W' + str(l + 1)] -= learning_rate * grads['dW' + str(l + 1)]
        parameters['b' + str(l + 1)] -= learning_rate * grads['db' + str(l + 1)]

    return parameters


# last L_model
def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations):
    costs = []

    parameters = initialize_parameters_deep(layers_dims)

    for iter in range(num_iterations):
        # forward
        AL, caches = L_model_forward(X, parameters)
        # compute cost
        cost = compute_cost(AL, Y)
        # backward
        grads = L_model_backward(AL, Y, caches)
        # update parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        if iter % 100 == 0:
            # print(AL)
            costs.append(cost)
            print('iter {}, cost: {}'.format(iter, cost))

    return parameters


def predict(X, parameters, show_accuracy=True):
    prediction, _ = L_model_forward(X, parameters)

    for i in range(0, len(prediction[0])):
        if prediction[0][i] >= 0.5:
            prediction[0][i] = 1
        else:
            prediction[0][i] = 0

    # if show_accuracy:
    #     print("Accuracy:", (np.dot(Y, prediction.T)+np.dot(1-Y, (1-prediction).T))/float(Y.size)*100, "%")
    return prediction


def image2prediction(my_image, parameters):
    fname = 'images/' + my_image
    im = Image.open(fname)
    im_reverse = im.resize((64, 64))
    my_image = np.asarray(im_reverse).reshape((1, 64*64*3)).T
    my_predicted_image = predict(my_image, parameters)

    plt.imshow(im)
    print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" +
          classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") + "\" picture.")
    plt.title(classes[int(np.squeeze(my_predicted_image)),].decode("utf-8"))
    plt.show()


if __name__ == '__main__':
    layers_dims = [12288, 20, 7, 5, 1]
    parameters = L_layer_model(train_x, train_y, layers_dims, learning_rate=0.0075,
                               num_iterations=2500)

    with open('./result/6.18_res.pickle', "wb") as fp:
        pickle.dump(parameters, fp, protocol=pickle.HIGHEST_PROTOCOL)

    with open('./result/6.18_res.pickle', "rb") as fp:
        parameters = pickle.load(fp)

    image2prediction('baidu_dog_1.jpeg', parameters)
```
