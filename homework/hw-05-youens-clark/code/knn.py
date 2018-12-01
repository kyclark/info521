#!/usr/bin/env python3
"""
Author : kyclark
Date   : 2018-11-24
Purpose: K-Nearest Neighbors
"""

import argparse
import matplotlib
import os
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import sys
from collections import Counter
from matplotlib.colors import ListedColormap
from scipy.spatial import distance


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

    parser.add_argument(
        '-k',
        metavar='INT',
        help='Values for K',
        nargs='+',
        type=int,
        default=[1, 5, 10, 59])

    parser.add_argument(
        '-o',
        '--out_dir',
        help='Output directory for saved figures',
        metavar='DIR',
        type=str,
        default=None)

    parser.add_argument(
        '-g',
        '--granularity',
        help='Granularity',
        metavar='int',
        type=int,
        default=100)

    parser.add_argument(
        '-q', '--quiet', help='Do not show figures', action='store_true')

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

    d = np.argsort(list(map(lambda z: distance.euclidean(p, z), x)))[:k]
    count = Counter(t[d])

    # most_common() returns a sorted list of tuples
    # so take the first element of the first tuple -- [0][0]
    return count.most_common(1)[0][0]


# --------------------------------------------------
def plot_decision_boundary(k, x, t, granularity=100, out_file=None):
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

    # Create color map
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
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

    if out_file:
        warn('Saving figure to "{}"'.format(out_file))
        plt.savefig(out_file)


# --------------------------------------------------
def main():
    args = get_args()
    in_file = args.file
    K = args.k
    granularity = args.granularity
    out_dir = args.out_dir

    if not os.path.isfile(in_file):
        die('"{}" is not a file'.format(in_file))

    x, t = read_data(in_file)

    basename, _ = os.path.splitext(os.path.basename(in_file))

    if out_dir:
        out_dir = os.path.abspath(out_dir)

    # Loop over different neighborhood values K
    for k in K:
        out_file = None
        if out_dir:
            out_file = os.path.join(out_dir, '{}-k-{}.png'.format(basename, k))
        plot_decision_boundary(
            k, x, t, granularity=granularity, out_file=out_file)

    if not args.quiet:
        warn('Showing figures')
        plt.show()

    warn('Done')


# --------------------------------------------------
if __name__ == '__main__':
    main()
