#!/usr/bin/env python3
"""
Author : Ken Youens-Clark <kyclark@email.arizona.edu>
Date   : 2018-11-24
Purpose: Cross-validation of K-Nearest Neighbors
"""

import argparse
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.spatial import distance
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split


# --------------------------------------------------
def get_args():
    """get command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Cross-validation of K-Nearest Neighbors',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '-f',
        '--file',
        help='Input file',
        metavar='FILE',
        type=str,
        default='../data/knn_binary_data.csv')

    parser.add_argument(
        '-i',
        '--iterations',
        help='How many iterartions for CV',
        metavar='int',
        type=int,
        default=10)

    parser.add_argument(
        '-o',
        '--outfile',
        help='File name to write figure',
        metavar='FILE',
        type=str,
        default='')

    parser.add_argument(
        '-k', help='Max value for K', metavar='int', type=int, default=30)

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
    return count.most_common(1)[0][0]


# --------------------------------------------------
def main():
    """Make a jazz noise here"""
    args = get_args()
    infile = args.file
    K = args.k
    iterations = args.iterations
    out_file = args.outfile

    df = pd.read_csv(infile)
    X = df.iloc[:, 0:2].values
    t = pd.to_numeric(df.iloc[:, 2].values, downcast='integer')

    kf = KFold(n_splits=iterations)

    x_plot = []
    y_plot = []
    for k in range(1, K + 1):
        predictions = []
        for train, test in kf.split(X):
            X_train, X_test, y_train, y_test = X[train], X[test], t[train], t[
                test]
            predicted = list(
                map(lambda p: knn(p, k, X_train, y_train), X_test))
            predictions.append(np.mean(predicted == y_test))

        print('k {} = {}'.format(k, np.mean(predictions)))
        x_plot.append(k)
        y_plot.append(np.mean(predictions))

    best = np.argmax(y_plot) + 1

    plt.plot(x_plot, y_plot)
    plt.title('Best k = {}'.format(best))

    if out_file:
        warn('Saving figure to "{}"'.format(out_file))
        plt.savefig(out_file)

    if not args.quiet:
        warn('Showing figure')
        plt.show()

    warn('Done')

# --------------------------------------------------
if __name__ == '__main__':
    main()
