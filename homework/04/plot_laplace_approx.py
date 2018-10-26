#!/usr/bin/env python3
"""
Author : kyclark
Date   : 2018-10-25
Purpose: Plot Laplace estimation of beta distribution
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
        description='Plot Laplace estimation of beta distribution',
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
        '-n',
        '--num_samples',
        help='n value',
        metavar='int',
        type=int,
        default=20)

    parser.add_argument(
        '-y', '--num_y', help='y value', metavar='int', type=int, default=10)

    parser.add_argument(
        '-o',
        '--outfile',
        help='Output file',
        metavar='str',
        type=str,
        default=None)

    parser.add_argument(
        '-d', '--debug', help='Talk about (pop music)', action='store_true')

    parser.add_argument(
        '-N', '--no_viz', help='Do not show pictures', action='store_true')

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
    """main"""
    args = get_args()
    alpha_val = args.alpha
    beta_val = args.beta
    n_val = args.num_samples
    y_val = args.num_y
    out_file = args.outfile

    def debug(msg):
        if args.debug:
            warn(msg)

    debug('α = {} β = {} N = {} y = {}'.format(alpha_val, beta_val, n_val,
                                               y_val))
    r_hat = (1 - y_val - alpha_val) / (2 - alpha_val - n_val - beta_val)
    debug('r_hat = {}'.format(r_hat))

    t1 = (y_val + alpha_val - 1) / np.power(r_hat, 2)
    t2 = (n_val - y_val + beta_val - 1) / np.power((1 - r_hat), 2)
    d2 = (-1 * t1) - t2
    sigma = -1 / d2
    debug('d2 = {}'.format(d2))
    debug('sigma = {}'.format(sigma))

    beta_dist = beta(alpha_val, beta_val)
    x = np.linspace(0, 1, n_val)

    plt.figure()
    plt.plot(x, beta_dist.pdf(x), 'blue')

    debug(norm.pdf(x, loc=r_hat, scale=np.sqrt(sigma)))
    plt.plot(x, norm.pdf(x, loc=r_hat, scale=np.sqrt(sigma)), 'red')

    plt.xlabel('x')
    plt.ylabel('y')

    tmpl = r'Laplace approx for $\alpha$ = {}, $\beta$ = {}, N = {}, y = {}'
    title = tmpl.format(alpha_val, beta_val, n_val, y_val)
    plt.title(title)

    if out_file:
        ext = '.png'
        if not out_file.endswith(ext):
            out_file += ext

        plt.savefig(out_file, fmt='png')

    if not args.no_viz:
        plt.show()


# --------------------------------------------------
if __name__ == '__main__':
    main()
