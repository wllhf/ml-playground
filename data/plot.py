import matplotlib
matplotlib.use('GTK3Agg')
# matplotlib.use('GTK3Cairo')
import matplotlib.pyplot as plt


def scatter_data(data, labels):
    plt.scatter(data[:, 0], data[:, 1], c=labels, s=50)


def clear_draw_and_wait(data, labels, marker=None):
    plt.clf()
    plt.scatter(data[:, 0], data[:, 1], c=labels, s=50)
    if marker is not None:
        plt.scatter(marker[:, 0], marker[:, 1], c='k', s=100)
    plt.draw()
    plt.waitforbuttonpress()
