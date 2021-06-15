# If you have 10,000,000 examples, how would you split the train/dev/test set?
- 98% train . 1% dev . 1% test                        √
- 33% train . 33% dev . 33% test
- 60% train . 20% dev . 20% test

# The dev and test set should:
- Come from the same distribution                     √
- Come from different distribution
- Be identical to each other (same(x, y) pairs)

# If your Neural Network model seems to have high variance, what of the following would be promising things to try?
- Add regularization                                  √ 
- Make the Neural Network deeper
- Increase the number of units in each hidden layer
- Get more test data
- Get more training data

# You are working on an automated check-out kiosk for a supermarket, and are building a classifier for apples, bananas and oranges. Suppose your classifier obtains a training set error of 0.5%, and a dev set error of 7%. Which of the following are promising things to try to improve your classifier? (Check all that apply.)
- Increase the regularization parameter lambda        √
- Decrease the regularization parameter lambda
- Get more training data                              √
- Use a bigger neural network

# What is weight decay?
- A technique to avoid vanishing gradient by imposing a ceiling on the values of the weights.
- A regularization technique (such as L2 regularization) that results in gradient descent shrinking the weights on every iteration.             √
- The process of gradually decreasing the learning rate during training.
- Gradual corruption of the weights in the neural network if it is trained on noisy data.

# What happens when you increase the regularization hyperparameter lambda?
- Weights are pushed toward becoming smaller (closer to 0)                                    √
- Weights are pushed toward becoming bigger (further from 0)
- Doubling lambda should roughly result in doubling the weights
- Gradient descent taking bigger steps with each iteration (proportional to lambda)           √

continue...









