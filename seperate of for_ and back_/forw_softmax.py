import numpy as np
def forw_softmax(x):
    x = np.array(x)
    y = np.zeros_like(x, dtype=np.float64)
    row = np.shape(x)[0]
    sum4ex = np.ones(1,np.float64)
    for i in range(0,row):
        ex = np.exp(x[i][0])
        sum4ex += ex
    for j in range(0,row):
        ex = np.exp(x[j][0])
        y[j][0] = ex / sum4ex
    return y

y = forw_softmax([[1],[2],[3]])