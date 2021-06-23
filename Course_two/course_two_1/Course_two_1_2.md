# Optimization

## 1. Mini-batch gradient descent
- if tranining set is way too big, so we can seperated into group called mini-batch, (baby training set)
- 1 epoch of training travel all training set

## 2. stochastic
- mini-batch shrink to just one training set each time
- a huge disadvantage is not use the speed up from vertorization

## 3. exponentially weighted averages
![image](https://user-images.githubusercontent.com/71109255/123089504-9ad8a800-d459-11eb-8498-582cea363d4e.png)

## 4. bias correction
- when you warm up your estimate, is not pretty well when is started. so you can use bias correction
- vt = vt / (1 - beta**2)

## Momentum
- basic idea: use exponentially weight averages to gradient descent
- ![image](https://user-images.githubusercontent.com/71109255/123090817-2b63b800-d45b-11eb-97ab-43ef3524e830.png)

## RMSprop: root mean square prop
- 
