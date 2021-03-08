import matplotlib.pyplot as plt
import numpy as np
import random as rd

dim = 100


def reward(i, j):
    string = dim/2
    bp = complex(dim/2, dim/2)
    bp = (bp.real, bp.imag)
    cp = complex(i, j)
    cp = (cp.real, cp.imag)
    xdiff = cp[0]-bp[0]
    if bp[1] < cp[1]:
        s = 1/(1 + 10*abs(bp[0]-cp[0])/string)
        # s = -(xdiff*xdiff-string*string)/string/string
    else:
        s = (abs((cp[0]-bp[0]))-string)/string
        # s = (xdiff*xdiff-string*string)/string/string
    return s


def draw_reward():

    image = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            image[i, j] = reward(j, dim-i)
    print(image)

    implot = plt.imshow(image, cmap='hot', vmin=-1, vmax=1)
    plt.show()


draw_reward()
