import numpy as np
def forw_softmax(x):
    # compute the value of e^x
    # compute the sum of e^x as sum4ex
    # compute the result of softmax = e^x/sum4ex
    x = np.array(x, dtype=np.float128)
    y = np.zeros_like(x, dtype=np.float128)
    row = np.shape(x)[0]
    sum4ex = np.ones(1, np.float128)
    for i in range(0, row):
        ex = np.exp(x[i][0])
        sum4ex += ex
    for j in range(0, row):
        ex = np.exp(x[j][0])
        y[j][0] = ex / sum4ex
    return y

def get_grad(y):
    # gradient matrix of softmax layer-dydx
    # use the chain rule for derivation,
    # find the derivative of forward propagation to the input
    # dyidxj = -yiyj, if i≠j
    # dyidxj = yi(1-yi), if i=j
    grad = y[:, np.newaxis] * y[np.newaxis, :]
    for i in range(len(grad)):
        grad[i, i] -= y[i]
    grad = - grad
    return grad

def back_softmax(x,y,dzdy):
    # backward propagation of softmax layer-dzdx
    # dzdxi = (dyidx1 + dyidxj +……+ dyidxn)*dzdy
    grad = get_grad(y)
    dzdx = np.sum(grad * dzdy, axis=1)
    return dzdx


#test
# make up some numbers for dz/dY coming in from backprop
dzdy = [[100],[200],[300]]
# make up some numbers for input array X
x = [[-1000],[-2000],[-3000]]
# forward pass to compute Y
y = forw_softmax(x)
# computing the backprop derivatives analytically
dzdx = back_softmax(x,y,dzdy)
# compute derivatives by using numerical derivatives
# numerically compute dz/dx
n=np.size(x,0)
eps = 1.0e-7
dzdxnumeric = np.zeros_like(dzdx)
newim = np.array(x, dtype=float)
for j in range(n):
    newim[j] = newim[j] + eps
    yprime = forw_softmax(newim)
    deriv = (yprime-y) / eps
    dzdxnumeric[j] = np.sum(deriv*dzdy)
    newim[j] = newim[j] - eps
# by making contract between dzdxnumeric and dzdx, if the results come almost the same. Correct
print("numerical derivatives")
print(dzdxnumeric)
print("numerical backprop derivatives")
print(dzdx)