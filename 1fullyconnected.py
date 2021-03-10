import numpy as np
def forw_fc(x,w,b):
    # input x and w are both M*N matrix
    # bias is a single number
    # fully_connected function realizes(the sum of dot(wij,xij)) + bias
    w = np.array(w, dtype=np.float64)
    x = np.array(x,dtype=np.float64)
    b = float(b)
    y = np.sum(w*x)+b
    return y

def back_fc(x,w,b,y,dzdy):
    w = np.array(w, dtype=np.float64)
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    b = float(b)
    dzdx = dzdy * w
    dzdw = dzdy * x
    dzdb = dzdy * 1
    return dzdx,dzdw,dzdb


#test
# set up number for dz/dy from backprop
dzdy = 5
# set up some numbers for input array x, w, b
x = [[1, 2, 6], [3, 4, 5], [1, 7, 3]]
w = [[1, 2, 3], [3, 3, 5], [1, 2, 6]]
b = 3
# forward pass to compute y by using forw_fc
y = forw_fc(x,w,b)
# computing the backprop derivatives
dzdx,dzdw,dzdb = back_fc(x,w,b,y,dzdy)
# compute derivatives by using numerical derivatives
# numerically compute dz/dx
eps = 1.0e-7
dzdxnumeric = np.zeros_like(dzdx)
newim = np.array(x, dtype=float)
width,height = np.array(newim, dtype=int).shape
for i in range(width):
    for j in range(height):
        newim[i, j] = newim[i, j] + eps
        yprime  = forw_fc(newim,w,b)
        deriv = (yprime-y) / eps
        dzdxnumeric[i, j] = np.sum(deriv*dzdy)
        newim[i, j] = newim[i, j] - eps
# by making contract between dzdxnumeric and dzdx, if the results come almost the same. Correct
print("numerically compute dz/dx")
print(dzdxnumeric)
print("backprop numerically compute dz/dx")
print(dzdx)

# compute derivatives by using numerical derivatives
# numerically compute dz/dx
eps = 1.0e-7
dzdwnumeric = np.zeros_like(dzdw)
newim2 = np.array(w, dtype=float)
width,height = np.array(newim2, dtype=int).shape
for i in range(width):
    for j in range(height):
        newim2[i, j] = newim2[i, j] + eps
        yprime  = forw_fc(x,newim2,b)
        deriv = (yprime-y) / eps
        dzdwnumeric[i, j] = np.sum(deriv*dzdy)
        newim2[i, j] = newim2[i, j] - eps
# by making contract between dzdwnumeric and dzdw, if the results come almost the same. Correct
print("numerically compute dz/dw")
print(dzdwnumeric)
print("backprop numerically compute dz/dw")
print(dzdw)

# compute derivatives by using numerical derivatives
# numerically compute dz/db
eps = 1.0e-7
newim3 = b
width,height = np.array(x, dtype=int).shape
for i in range(width):
    for j in range(height):
        newim3 = newim3 + eps
        yprime  = forw_fc(x,w,newim3)
        deriv = (yprime-y) / eps
        dzdbnumeric = np.sum(deriv*dzdy)
        newim3 = newim3 - eps
# by making contract between dzdbnumeric and dzdb, if the result comes almost the same. Correct
print("numerically compute dz/db")
print(dzdbnumeric)
print("backprop numerically compute dz/db")
print(dzdb)
