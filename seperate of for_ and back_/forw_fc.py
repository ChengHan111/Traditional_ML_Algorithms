import numpy as np
def forw_fc(x,w,b):
    w = np.array(w)
    x = np.array(x)
    b = float(b)
    y = np.sum(w*x)+b
    return y

y = forw_fc([[1,3],
                [2,4]],
                [[1,2],
                [6,7]],
                        3)