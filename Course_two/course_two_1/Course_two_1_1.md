# 1.1 Train/dev/test sets
- normal data use 60/20/20 % for train/dev/test set
- when data is more than 1000000, you can use 98/1/1 % for train/dev/test set

# bias / variance
- high bias, underfitting.
- high variance, overfitting.
- just right, right bias and variance
- through Train set error and Dev set error to see high variance or high bias or not
![image](https://user-images.githubusercontent.com/71109255/122025153-b4477780-cdfb-11eb-9b03-44b6e99d50dc.png)

# basic recipe for machine learning
![image](https://user-images.githubusercontent.com/71109255/122026034-7565f180-cdfc-11eb-8984-8cf25dc4b015.png)

# Regularization
- L1 and L2
![image](https://user-images.githubusercontent.com/71109255/122027259-8cf1aa00-cdfd-11eb-8025-a118c6cffec9.png)
- dropout
![image](https://user-images.githubusercontent.com/71109255/122034615-46537e00-ce04-11eb-9169-e1c19c703241.png)
![image](https://user-images.githubusercontent.com/71109255/122035315-edd0b080-ce04-11eb-89a8-c35264de82a3.png)
the essence of dropout is that when you dropout some unit, like 20%, you must make next layer's a not be decrease, so make previous layer's a divide by 80%
- flipping your pic to make 1 pic --> 2 pic
- early stopping training, just look the follow picture
![image](https://user-images.githubusercontent.com/71109255/122066754-968f0800-ce25-11eb-9d38-71163803ae78.png)

# speeding up your training
- normalizing inputs into 0 mean and 1 variance
- sigma ** 2 = 1/m\*sum(x(i) ** 2), and x /= sigma ** 2
![image](https://user-images.githubusercontent.com/71109255/122072023-cfc97700-ce29-11eb-8e9c-7ae90b82c144.png)

# gradient vanish or explode
- ok i got it, if have 150 layers deep neurual network, if b[l] = 0, w[l] = 1.1E, so output will explode, if 0.9E, output will be vanish

# initialize weight to reduce explode or vanish
![image](https://user-images.githubusercontent.com/71109255/122074254-a1e53200-ce2b-11eb-81e3-f4b14f4926ed.png)
- different activation function use different ways to initialized weight

# gradient checking
- Take W[1], b[1], ... , W[l], b[l] reshape and into a big vector W
- Take dW[1], db[1], ... , dW[l], db[l] reshape and into a big vector DW
