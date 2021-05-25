# 2.1 binary classification
### some tips
1. if had m training set, it usually don't use for loop.
2. it have forward propagation step and backward propagation step.
### example logistic regression cat classification
1. 1(cat) vs 0(non cat)
2. if input image is 64x64 pixels RGB image, then it will be **64x64x3** pixel intensity values into features vector x = [col(64x64x3=12288)]
3. nx = 12288
4. x --> y
### notation
![](images/2.1 notation.png)
- (x, y), x∈Rnx, y∈{0, 1}
- m, traning example: {(x1, y1), (x2, y2),...,(xm, ym)}
- X = [x1, x2,..., xm], (12288 x m matrix)
- Y = [y1, y2,..., ym], (1 x m matrix)

# 2.2 logistic regression use notation
### 
- Given x, want y_preidct = P(y=1|x), means y_predict [0, 1], so relu or sigmod will be use
- Parameters: **W**, **b**
- Output: y_predict = sigmod(W_T**x**+b)

# 2.3 cost funtion in logistic regression
- Loss (error) function: L(y_pred, y) = -(ylogy_pred + (1-y)log(1-y_pred)), prove: if you let y = 1, that means you want y_pred as big as possible but no bigger than 1, because sigmod function
- Cost function: J(w, b) = 1/m x sum(L(y_pred, y))
- Loss function is represent one single training sample, Cost function in represent all training good or not. 

# 2.4 how to use gradient desent to train w and b
### example
- let's do a J(w) (simple function like y = x^2)'s gradient desent.
- repeat { w:= w - α(dJ(w)/dw)}, α: learning rate
