import numpy as np
def back_relu(x,y,dzdy):
    # define back_relu function, which is the backward propagation of relu layer
    # dzdx = dzdy * dydx
    # dydx = 1 if x>0
    # dydx = 0 if x<0
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    dzdy = np.array(dzdy)
    l = np.maximum(x, 0)
    dzdx = dzdy*(l/x)
    return dzdx
# make up some numbers for input array X
x = [[-5, -6, 7, 8]]

y = [[0,0,7,8]]
# make up some numbers for dz/dY coming in from backprop
dzdy = [[1, 2, 3, 4]]

# computing the backprop derivatives analytically
dzdx = back_relu(x, y, dzdy)
# compute derivatives by using numerical derivatives
# numerically compute dz/dx
eps = 1.0e-6
dzdxnumeric = np.zeros_like(dzdx)
newim = np.array(y, dtype=float)
width,height = np.array(y, dtype=float).shape
for i in range(width):
    for j in range(height):
        newim[i, j] = newim[i, j] + eps
        yprime  = back_relu(x,newim,dzdy)
        deriv = (yprime-y) / eps
        dzdxnumeric[i, j] = np.sum(deriv*dzdy)
        newim[i, j] = newim[i, j] - eps
print(dzdxnumeric)
print(dzdx)


# dzdx = back_relu([[-10, 2, 3],
#                     [4, -1, 6],
#                     [7, -10, 9]],
#                          [[-10, 2, 3],
#                          [4, -1, 6],
#                          [7, -10, 9]],
#                             [[-10, 2, 3],
#                             [4, -1, 6],
#                             [7, -10, 9]])