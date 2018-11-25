#!/usr/bin/env python3
"""
Author : kyclark
Date   : 2018-11-24
Purpose: Cross-validation of K-Nearest Neighbors
"""

import argparse
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.spatial import distance
from sklearn.model_selection import ShuffleSplit


# --------------------------------------------------
def get_args():
    """get command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Cross-validation of K-Nearest Neighbors',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    #parser.add_argument(
    #    'positional', metavar='str', help='A positional argument')

    parser.add_argument(
        '-f',
        '--file',
        help='Input file',
        metavar='FILE',
        type=str,
        default='')

    parser.add_argument(
        '-i',
        '--iterations',
        help='How many iterartions for CV',
        metavar='int',
        type=int,
        default=10)

    parser.add_argument(
        '-k', help='K-nearest neighbors', metavar='int', type=int, default=30)

    #parser.add_argument(
    #    '-f', '--flag', help='A boolean flag', action='store_true')

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

    d = map(lambda z: distance.euclidean(p, z), x)

    count = Counter()
    for i, pos in enumerate(np.argsort(d)):
        target = t[pos]
        count[target] += 1
        if i == k:
            break

    # most_common() returns a sorted list of tuples
    # so take the first element of the first tuple -- [0][0]
    return count.most_common(1)[0][0]


# --------------------------------------------------
def main():
    """Make a jazz noise here"""
    args = get_args()
    df = pd.read_csv(args.file)
    X = df.iloc[:, 0:2].values
    t = df.iloc[:, 2].values

    # rs = ShuffleSplit(n_splits=args.iterations, test_size=0.3)
    # for i, (train_index, test_index) in enumerate(rs.split(X)):
    #     print("{}: TRAIN: {}\nTEST: {}\n".format(i+1, train_index, test_index))
    #     x = 

    for i in range(args.iterations):
        X_train, X_test, y_train, y_test = train_test_split(
                X, t, test_size=0.33)
        predicted = map(lambda p: knn(p, args.k, 
        for j in range(len(x_train)):
            knn(x_train[j], y_train, args.k,


# --------------------------------------------------
if __name__ == '__main__':
    main()
