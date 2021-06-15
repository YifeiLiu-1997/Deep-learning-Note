# 3.4 activation functions
- a = tanh(z) is better than a=sigmod(z)
- if y_pred belong [0,1], you can choose sigmod
- ReLU(rectified linear unit) activation function is more better
- Leaky ReLU 
- all this activation function just doing one thing: let gradient desent's derivative far away from zero

# 3.11 random initialization
- zero init is trash, because all activation value is the same
- usually use small random weight init, because it can make sigmod(z) not in soft

