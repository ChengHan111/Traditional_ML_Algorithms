import numpy as np
def forw_meanpool(x):
    # define the forw_meanpool function
    x = np.array(x)
    height = np.shape(x)[0]
    width = np.shape(x)[1]
    # known it is a 2x2 filter to do the mean function, this size can change
    w_height = 2
    w_width = 2
    # for different situation when x is a matrix with odd r or l or both
    final_height = int((height - w_height) / 2) + 1
    final_width = int((width - w_width) / 2) + 1
    y = np.zeros((final_height, final_width), dtype=np.float64())

    for k in range(final_height):
        for m in range(final_width):
            start_k = k * 2
            start_m = m * 2
            end_k = start_k + w_height
            end_m = start_m + w_width

            y[k, m] = np.mean(x[start_k: end_k, start_m: end_m])
            # compute the mean value
    return y

def back_meanpool(x,y,dzdy):
    # define back_meanpool function
    # input x,y,dzdy where dzdy is the loss of backward propagation
    # output is dzdx
    # dzdx = dzdy*dydx
    # dydx = 1 / (2x2) in this situation
    x = np.array(x)
    y = np.array(y)
    dzdy = np.array(dzdy)
    dzdx = np.zeros_like(x, dtype=np.float64)

    height = np.shape(dzdy)[0]
    width = np.shape(dzdy)[1]
    w_height = 2
    w_width = 2

    for i in range(height):
        for j in range(width):
            start_i = i * 2
            start_j = j * 2
            end_i = start_i + w_height
            end_j = start_j + w_width

            dzdx[start_i: end_i, start_j: end_j] = dzdy[i,j] / 4
    return dzdx


#test
# make up some numbers for dz/dY coming in from backprop
dzdy = [[1, 6],[2, 4]]
dzdy = np.array(dzdy)
# make up some numbers for input array X
x = [[1, 2, 3, 4],[3, 4, 5, 6],[1, 2, 3, 4],[3, 4, 5, 6]]
# forward pass to compute Y
y = forw_meanpool(x)
# computing the backprop derivatives analytically
dzdx = back_meanpool(x, y, dzdy)
# compute derivatives by using numerical derivatives
# numerically compute dz/dx
width, height = np.array(x, dtype=int).shape
eps = 1.0e-6
dzdxnumeric = np.zeros_like(dzdx)
newim = np.array(x, dtype=float)
for i in range(width):
    for j in range(height):
        newim[i, j] = newim[i, j] + eps
        yprime  = forw_meanpool(newim)
        deriv = (yprime-y) / eps
        dzdxnumeric[i, j] = np.sum(deriv*dzdy)
        newim[i, j] = newim[i, j] - eps
# by making contract between dzdxnumeric and dzdx, if the results come almost the same. Correct
print("numerical derivatives")
print(dzdxnumeric)
print("numerical backprop derivatives")
print(dzdx)
