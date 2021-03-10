import numpy as np
def back_maxpool(x,y,dzdy):
    x = np.array(x)
    y = np.array(y)
    dzdy = np.array(dzdy)
    dydx = np.zeros_like(x)

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

    for i in range(final_height):
        for j in range(final_width):
            start_i = i * 2
            start_j = j * 2
            end_i = start_i + w_height
            end_j = start_j + w_width

            index = np.unravel_index(position_detect[i, j], (2, 2))
            dydx[start_i: end_i, start_j: end_j][index] = dzdy[i, j]
    return dydx

dydx = back_maxpool([[-10, 4, 3, 4],
                        [4, -1, 6, 7],
                         [-10, 2, 3, 4],
                         [4, -1, 6, 7]],
                            [[7, -10],
                                [1,12]],
                            [[-10, 2],
                                [4, -1]])
