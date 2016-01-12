import numpy as np


def generate_edge(dx, dy, alpha):
    img = np.zeros((dy, dx))
    mx, my = [(d-1)/2 for d in [dx, dy]]

    for i in range(-dx/2, dx/2):
        x = mx+i*np.sin(alpha)
        y = min(dy, my+i)
        img[y, x] = 1

    return img


if __name__ == '__main__':
    print generate_edge(5, 5, 0)

