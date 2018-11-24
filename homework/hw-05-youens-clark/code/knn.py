#!/usr/bin/env python3
"""
Author : kyclark
Date   : 2018-11-24
Purpose: K-Nearest Neighbors
"""

import argparse
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import sys
from collections import Counter
from matplotlib.colors import ListedColormap
from scipy.spatial import distance

# Create color map
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])


# --------------------------------------------------
def get_args():
    """get command-line arguments"""
    parser = argparse.ArgumentParser(
        description='K-Nearest Neighbors',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '-f',
        '--file',
        metavar='FILE',
        help='Input file',
        default='../data/knn_binary_data.csv')

    # parser.add_argument(
    #     '-a',
    #     '--arg',
    #     help='A named string argument',
    #     metavar='str',
    #     type=str,
    #     default='')

    # parser.add_argument(
    #     '-i',
    #     '--int',
    #     help='A named integer argument',
    #     metavar='int',
    #     type=int,
    #     default=0)

    # parser.add_argument(
    #     '-f', '--flag', help='A boolean flag', action='store_true')

    return parser.parse_args()


# --------------------------------------------------
def warn(msg):
    """Print a message to STDERR"""
    print(msg, file=sys.stderr)


# --------------------------------------------------
def die(msg='Something bad happened'):
    """warn() and exit with error"""
    warn(msg)
    sys.exit(1)


# --------------------------------------------------
def read_data(path, d=','):
    """
    Read 2-dimensional real-valued features with associated class labels
    :param path: path to csv file
    :param d: delimiter
    :return: x=array of features, t=class labels
    """
    arr = np.genfromtxt(path, delimiter=d, dtype=None)
    length = len(arr)
    x = np.zeros(shape=(length, 2))
    t = np.zeros(length)
    for i, (x1, x2, tv) in enumerate(arr):
        x[i, 0] = x1
        x[i, 1] = x2
        t[i] = int(tv)
    return x, t


# --------------------------------------------------
def knn(p, k, x, t):
    """
    K-Nearest Neighbors classifier.  Return the most frequent class among the k
    nearest points
    :param p: point to classify (assumes 2-dimensional)
    :param k: number of nearest neighbors
    :param x: array of observed 2-dimensional points
    :param t: array of target labels (corresponding to points)
    :return: the top class label
    """

    # Number of instances in data set
    N = x.shape[0]
    d = list(map(lambda z: distance.euclidean(p, z), x))

    count = Counter()
    for i, pos in enumerate(np.argsort(d)):
        target = t[pos]
        #print("i {} pos {} target {}".format(i, pos, target))
        count[target] += 1
        if i == k:
            break

    # most_common() returns a sorted list of tuples
    # so take the first element of the first tuple -- [0][0]
    return count.most_common(1)[0][0]


# --------------------------------------------------
def plot_decision_boundary(k, x, t, granularity=100):
    """
    Given data (observed x and labels t) and choice k of nearest neighbors,
    plots the decision boundary based on a grid of classifications over the
    feature space.
    :param k: number of nearest neighbors
    :param x: array of observed 2-dimensional points
    :param t: array of target labels (corresponding to points)
    :param granularity: controls granularity of the meshgrid
    :return:
    """
    print('KNN for K={0}'.format(k))

    # Initialize meshgrid to be used to store the class prediction values
    # this is used for computing and plotting the decision boundary contour
    Xv, Yv = np.meshgrid(
        np.linspace(np.min(x[:, 0]) - 0.1,
                    np.max(x[:, 0]) + 0.1, granularity),
        np.linspace(np.min(x[:, 1]) - 0.1,
                    np.max(x[:, 1]) + 0.1, granularity))

    # Calculate KNN classification for every point in meshgrid
    classes = np.zeros(shape=(Xv.shape[0], Xv.shape[1]))
    for i in range(Xv.shape[0]):
        for j in range(Xv.shape[1]):
            classes[i][j] = knn(np.array([Xv[i][j], Yv[i][j]]), k, x, t)

    # plot the binary decision boundary contour
    plt.figure()
    plt.pcolormesh(Xv, Yv, classes, cmap=cmap_light)
    ti = 'K = {0}'.format(k)
    plt.title(ti)
    plt.draw()

    # Plot the points
    ma = ['o', 's', 'v']
    fc = ['r', 'g', 'b']  # np.array([0, 0, 0]), np.array([1, 1, 1])]
    tv = np.unique(t.flatten())  # an array of the unique class labels

    #if new_figure:
    #    plt.figure()

    for i in range(tv.shape[0]):
        # returns a boolean vector mask for selecting just the instances of class tv[i]
        pos = (t == tv[i]).nonzero()
        plt.scatter(
            np.asarray(x[pos, 0]),
            np.asarray(x[pos, 1]),
            marker=ma[i],
            facecolor=fc[i])


# --------------------------------------------------
def main():
    args = get_args()
    x, t = read_data(args.file)

    # Loop over different neighborhood values K
    for k in [1, 5, 10, 59]:
        plot_decision_boundary(k, x, t)

    plt.show()


# --------------------------------------------------
if __name__ == '__main__':
    main()
