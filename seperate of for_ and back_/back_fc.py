import numpy as np
def back_fc(x,w,b,y,dzdy):
    w = np.array(w)
    x = np.array(x)
    y = np.array(y)
    b = float(b)
    dzdx = dzdy * w
    dzdw = dzdy * x
    dzdb = dzdy * 1
    return dzdx,dzdw,dzdb

dzdx,dzdw,dzdb = back_fc([[1,3,4],
                [2,4,5]],
                [[1,2,3],
                [6,7,8]],
                        3,
                    [[4,9,15],
                    [15,31,43]],
                                2)