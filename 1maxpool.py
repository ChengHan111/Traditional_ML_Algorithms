import numpy as np
def forw_maxpool(x):
    # define the forw_maxpool function
    x = np.array(x)
    height = np.shape(x)[0]
    width = np.shape(x)[1]
    # known it is a 2x2 filter to do the mean function, this size can change
    w_height = 2
    w_width = 2
    # for different situation when x is a matrix with odd r or l or both
    final_height = int((height - w_height) / 2) + 1
    final_width = int((width - w_width) / 2) + 1

    y = np.zeros((final_height,final_width), dtype=np.float64())

    for k in range(final_height):
        for m in range(final_width):

            start_k = k * 2
            start_m = m * 2

            end_k = start_k + w_height
            end_m = start_m + w_width

            y[k, m] = np.max(x[start_k: end_k, start_m: end_m])
            # compute the maximum value
    return y

def back_maxpool(x,y,dzdy):
    # Backward propagation of maxpool
    # input = dzdy
    # output = dzdx
    # dzdx = dzdy*dydx
    # dyidxj = 1 when xj is the maximun number
    # dyidxj = 0 when xj is not the maximun number
    x = np.array(x,dtype=float)
    y = np.array(y, dtype=float)
    dzdy = np.array(dzdy, dtype=float)
    dzdx = np.zeros_like(x, dtype=float)

    height = np.shape(x)[0]
    width = np.shape(x)[1]

    w_height = 2
    w_width = 2

    final_height = int((height - w_height) / 2) + 1
    final_width = int((width - w_width) / 2) + 1

    y = np.zeros((final_height,final_width), dtype=np.float64())

    position_detect = np.zeros_like(y, dtype=np.int32)

    for k in range(final_height):
        for m in range(final_width):
            start_k = k * 2
            start_m = m * 2
            end_k = start_k + w_height
            end_m = start_m + w_width

            y[k, m] = np.max(x[start_k: end_k, start_m: end_m])
            position_detect[k, m] = np.argmax(x[start_k: end_k, start_m: end_m])
            position_detect = position_detect
            # compute the maximum value
            # store the location of the maximum number in position_detect by using argmax
    for i in range(final_height):
        for j in range(final_width):
            start_i = i * 2
            start_j = j * 2
            end_i = start_i + w_height
            end_j = start_j + w_width

            index = np.unravel_index(position_detect[i, j], (2, 2))
            dzdx[start_i: end_i, start_j: end_j][index] = dzdy[i, j]
    return dzdx


#test
# make up some numbers for dz/dY coming in from backprop
dzdy = [[3, 6]]
# make up some numbers for input array X
x = [[1, 7, 3, 4],[3, 4, 10, 6]]
# forward pass to compute Y and location of maximum
y= forw_maxpool(x)
# computing the backprop derivatives analytically
dzdx = back_maxpool(x, y, dzdy)

# compute derivatives by using numerical derivatives
# numerically compute dz/dx
width, height = np.array(x, dtype=int).shape
eps = 1.0e-6
dzdxnumeric = np.zeros_like(dzdx)
newim = np.array(x, dtype=float)
for i in range(width):
    for j in range(height):
        newim[i, j] = newim[i, j] + eps
        yprime = forw_maxpool(newim)
        deriv = (yprime-y) / eps
        dzdxnumeric[i, j] = np.sum(deriv*dzdy)
        newim[i, j] = newim[i, j] - eps
# by making contract between dzdxnumeric and dzdx, if the results come almost the same. Correct
print("numerical derivatives")
print(dzdxnumeric)
print("numerical backprop derivatives")
print(dzdx)