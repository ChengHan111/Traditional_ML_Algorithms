import numpy as np
def forw_relu(x):
    # define the relu function
    x = np.array(x, dtype=float)
    # when x>0, output = x
    #when x<0, output = 0
    l = np.maximum(x, 0)
    return l
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
