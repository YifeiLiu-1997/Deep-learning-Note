# 4.1 notation
![image](https://user-images.githubusercontent.com/71109255/121980669-ef778580-cdbe-11eb-8564-0fcd402cdbfe.png)

l = 4              layers

n[l]               units in layer l

n[1] = 5, 

n[2] = 5, 

n[3] = 3, 

n[4] = 1, 

n[0] = input = 3

a[l] activations in layer l

a[l] = g[l]\(z[l])

w[l] weight in layer l

# 4.2 forward prop and backward prop
![image](https://user-images.githubusercontent.com/71109255/121987408-fe643500-cdca-11eb-99b6-887a15c743ce.png)

# 4.3 check dimension of matrix to see if your code is wrong or not
w[l].shape = (n[l], n[l-1])
b[l].shape = (n[l], 1)
dw[l].shape = w[l].shape
db[l].shape = b[l].shape

# 4.4 why deep neurual network work
for example, when you do a face recognize, 3-hidden layer represent edge, nose, face(just example)

# parameters, hyper parameters
parameters: W, b
Hyperparameters: learning rate alpha, iterations, hidden layers l, hidden units n[1], n[2], activation function
