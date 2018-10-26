#!/usr/bin/env python3
"""
Author : kyclark
Date   : 2018-10-25
Purpose: Plot Laplace estimation
"""

import argparse
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta, norm


# --------------------------------------------------
def get_args():
    """get args"""
    parser = argparse.ArgumentParser(
        description='Plot Laplace estimation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '-a',
        '--alpha',
        help='alpha value',
        metavar='int',
        type=int,
        default=5)

    parser.add_argument(
        '-b', '--beta', help='beta value', metavar='int', type=int, default=5)

    parser.add_argument(
        '-n', help='n value', metavar='int', type=int, default=20)

    parser.add_argument(
        '-y', help='y value', metavar='int', type=int, default=10)

    parser.add_argument(
        '-d', '--debug', help='Talk about (pop music)', action='store_true')

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
def laplace(alpha, beta, y, n):
    r_hat = (1 - y + alpha) / (n - (2 * y) - alpha + beta)
    print('r_hat = {}'.format(r_hat))
    t1 = ((1 - y - alpha) / np.power(r_hat, 2))
    t2 = (n - y + beta - 1) / np.power(r_hat - 1, 2)
    d2 = t1 - t2
    return r_hat, -1 * d2

# --------------------------------------------------
def main():
    """main"""
    args = get_args()
    alpha_val = args.alpha
    beta_val = args.beta
    n_val = args.n
    y_val = args.y

    beta_dist = beta(alpha_val, beta_val)
    x = np.linspace(0, 1, n_val)

    plt.figure()
    plt.plot(x, beta_dist.pdf(x))

    r_hat, sigma = laplace(alpha_val, beta_val, y_val, n_val)
    normal_dist = norm.pdf(x, loc=r_hat, scale=sigma)
    plt.plot(x, normal_dist.pdf(x))

    #mu = alpha_val / (alpha_val + beta_val)
    #plt.plot(mu, beta_dist.pdf(mu), 'bo')

    plt.xlabel('$\mu$')
    plt.ylabel('y')
    plt.title(r'Beta pdf for $\alpha$ = {} $\beta$ = {}'.format(
        alpha_val, beta_val))
    plt.show()


# --------------------------------------------------
if __name__ == '__main__':
    main()
