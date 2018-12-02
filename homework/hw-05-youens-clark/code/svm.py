#!/usr/bin/env python3
"""
Author : Ken Youens-Clark <kyclark@email.arizona.edu>
Date   : 2018-12-02
Purpose: Demonstrate affect on SVM of removing a support vector
"""

import argparse
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn import svm


# --------------------------------------------------
def get_args():
    """get command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Demonstrate affect on SVM of removing a support vector' ,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '-o',
        '--outfile',
        help='File to write figure',
        metavar='FILE',
        type=str,
        default=None)

    parser.add_argument(
        '-r',
        '--random_seed',
        help='Random seed value',
        metavar='int',
        type=int,
        default=None)

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
def main():
    """Make a jazz noise here"""
    args = get_args()

    fig, ax = plt.subplots(2)

    if args.random_seed is not None:
        np.random.seed(args.random_seed)

    X, y = make_blobs(n_samples=20, centers=2)

    clf = svm.SVC(kernel='linear', C=1000)
    clf.fit(X, y)

    #
    # Create grid to evaluate model
    #
    xlim = ax[0].get_xlim()
    ylim = ax[0].get_ylim()
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    #
    # Plot data, decision boundary and margins, support vectors
    #
    ax[0].scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

    ax[0].contour(
        XX,
        YY,
        Z,
        colors='k',
        levels=[-1, 0, 1],
        alpha=0.5,
        linestyles=['--', '-', '--'])

    ax[0].scatter(
        clf.support_vectors_[:, 0],
        clf.support_vectors_[:, 1],
        s=100,
        linewidth=1,
        facecolors='none',
        edgecolors='k')

    #
    # Now remove one of the supports and refit
    #
    support = clf.support_
    X2 = np.delete(X, support[0], axis=0)
    y2 = np.delete(y, support[0])

    clf.fit(X2, y2)

    Z2 = clf.decision_function(xy).reshape(XX.shape)

    ax[1].scatter(X2[:, 0], X2[:, 1], c=y2, s=30, cmap=plt.cm.Paired)

    ax[1].contour(
        XX,
        YY,
        Z2,
        colors='k',
        levels=[-1, 0, 1],
        alpha=0.5,
        linestyles=['--', '-', '--'])

    ax[1].scatter(
        clf.support_vectors_[:, 0],
        clf.support_vectors_[:, 1],
        s=100,
        linewidth=1,
        facecolors='none',
        edgecolors='k')

    if args.outfile:
        warn('Saving figure to "{}"'.format(args.outfile))
        plt.savefig(args.outfile)

    plt.show()


# --------------------------------------------------
if __name__ == '__main__':
    main()
