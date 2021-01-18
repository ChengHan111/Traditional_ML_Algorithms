import numpy as np
# 当Feature数量小于100000时使用Normal Equation；
# 当Feature数量大于100000时使用Gradient Descent；

#let x be shape: (training example, features)
# y be (training example, 1)
# Output w (features, 1)

def linear_regression_normal_equation(X,y):
    ones = np.ones((X.shape[0], 1))
    X = np.append(ones, X, axis=1) # for bias
    W = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, y))
    return W
if __name__ == '__main__':
    X = np.array([[0, 1, 2]]).T
    y = np.array([-1, 0, 1])
    print(X.shape)
    print(y.shape)
    W = linear_regression_normal_equation(X, y)
    print(W)

    X1 = np.random.rand(5000, 1)
    y1 = 5*X1 + np.random.rand(5000, 1)* 0.1
    W1 = linear_regression_normal_equation(X1, y1)
    print(W1)