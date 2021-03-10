import numpy as np
def forw_relu(x):
    # define the relu function
    x = np.array(x, dtype=float)
    # when x>0, output = x
    # when x<0, output = 0
    l = np.maximum(x, 0)
    return l

def back_relu(x,y,dzdy):
    # define back_relu function, which is the backward propagation of relu layer
    # dzdx = dzdy * dydx
    # dydx = 1 if x>0
    # dydx = 0 if x<0
    y = np.array(y, dtype=float)
    dzdy = np.array(dzdy, dtype=float)
    l = np.maximum(x, 0)
    dzdx = dzdy*(y>0)
    return dzdx


#test
# set x as the input and do the test
x = [[-7, -3, 7, 8]]
# set y as the output of forw_relu(also y is known in back_relu funtion, which should be the same)
y = forw_relu(x)
# set numbers for dz/dY when doing backprop
dzdy = [[1, 3, 5, 4]]
dzdy = np.array(dzdy,dtype=float)
# computing the backprop derivatives by using the back_relu function
dzdx = back_relu(x, y, dzdy)
# compute derivatives by using numerical derivatives
# compute dz/dx
eps = 1.0e-6
dzdxnumeric = np.zeros_like(dzdx)
newim = np.array(x, dtype=float)
width,height = np.array(y, dtype=int).shape
for i in range(width):
    for j in range(height):
        newim[i, j] = newim[i, j] + eps
        yprime  = forw_relu(newim)
        deriv = (yprime-y) / eps
        dzdxnumeric[i, j] = np.sum(deriv*dzdy)
        newim[i, j] = newim[i, j] - eps
# by making contract between dzdxnumeric and dzdx, if the results come almost the same. Correct
print("numerical derivatives")
print(dzdxnumeric)
print("numerical backprop derivatives")
print(dzdx)