#!/usr/bin/env python3
"""docstring"""

# approx_expected_value_sin.py
# Port of approx_expected_value.m
# From A First Course in Machine Learning, Chapter 2.
# Simon Rogers, 01/11/11 [simon.rogers@glasgow.ac.uk]
# Approximating expected values via sampling
# Ken Youens-Clark 5 Oct 2018

import argparse
import matplotlib.pyplot as plt
import numpy as np
import sys


# --------------------------------------------------
def get_args():
    """get args"""
    parser = argparse.ArgumentParser(
        description='Approximate Expected Value',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '-s', '--save', help='Save outputs', action='store_true')

    return parser.parse_args()


# --------------------------------------------------
def main():
    """main"""
    args = get_args()
    save_figs = args.save

    # Plot the function!
    plot_fn(y_fn, -4, 10, y_fn_name, save_fig=args.save)

    # Now we'll approximate the area under the curve using sampling...

    # Sample 5000 uniformly random values in [-4..10]
    xs = np.random.uniform(low=-4.0, high=10.0, size=5000)
    ys = y_fn(xs)

    # compute the expectation of y, where y is the function that squares its
    # input
    ey2 = np.mean(ys)
    print('Sample-based approximation: {:f}'.format(ey2))

    #print('\int_y(-4) = {}'.format(int_y_fn(-4)))
    #print('\int_y(10) = {}'.format(int_y_fn(10)))
    e_fn = (int_y_fn(10) - int_y_fn(-4)) / 14
    print('Calculated expectation    : {:f}'.format(e_fn))

    # Store the evolution of the approximation, every 10 samples
    sample_sizes = np.arange(1, ys.shape[0], 10)
    ey2_evol = np.zeros(
        (sample_sizes.shape[0]))  # storage for the evolving estimate...
    # the following computes the mean of the sequence up to i, as i iterates
    # through the sequence, storing the mean in ey2_evol:
    for i in range(sample_sizes.shape[0]):
        ey2_evol[i] = np.mean(ys[0:sample_sizes[i]])

    # Create plot of evolution of the approximation
    plt.figure()
    # plot the curve of the estimation of the expected value of f(x)=y^2
    plt.plot(sample_sizes, ey2_evol)

    # The true, analytic result of the expected value
    # plot the analytic expected result as a red line:
    plt.plot(
        np.array([sample_sizes[0], sample_sizes[-1]]),
        np.array([ey2, ey2]),
        color='r')
    plt.xlabel('Sample size')
    plt.ylabel('Approximation of expectation')
    plt.title('{} ~= {:.02f}'.format(y_fn.__doc__, ey2))
    plt.pause(.1)  # required on some systems so that rendering can happen

    if args.save:
        outfile = 'approx.png'
        print('Saving to "{}"'.format(outfile))
        plt.savefig(outfile, fmt='png')

    plt.show()  # keeps the plot open

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
def plot_fn(fn, a, b, fn_name, resolution=100, save_fig=False):
    x = np.append(np.array([a]), np.append(np.linspace(a, b, resolution), [b]))
    font = {
        'family': 'serif',
        'color': 'darkred',
        'weight': 'normal',
        'size': 12,
    }
    y = fn(x)
    y[0] = 0
    y[-1] = 0
    plt.figure()
    plt.fill(x, y, 'b', alpha=0.3)
    fname, x_tpos, y_tpos = fn_name()
    plt.text(x_tpos, y_tpos, fname, fontdict=font)
    plt.title('Area under function')
    plt.xlabel('$y$')
    plt.ylabel('$f(y)$')

    if save_fig:
        outfile = 'approx_area_under_f.png'
        print('Saving to "{}"'.format(outfile))
        plt.savefig(outfile, fmt='png')

    x_range = b - a
    plt.xlim(a - (x_range * 0.1), b + (x_range * 0.1))


# --------------------------------------------------
# Define the function y that we are going to plot
def y_fn(x):
    """$t = 35x+3x-0.5x^3+0.05x^4$"""
    return 35 + (3 * x) - (0.5 * np.power(x, 3)) + (0.05 * np.power(x, 4))


# --------------------------------------------------
def int_y_fn(x):
    """
    This is a hard-coded version of the integral I used to check my arithmetic
    :return: float
    """
    return (35 * x) + (1.5 * np.power(x, 2)) - ((0.5 / 4) * np.power(x, 4)) + (
        (0.05 / 5) * np.power(x, 5))


# --------------------------------------------------
def y_fn_name():
    """
    Helper for displaying the name of the fn in the plot
    Returns the parameters for plotting the y function name,
    used by approx_expected_value
    :return: fname, x_tpos, y_tpos, year
    """
    fname = y_fn.__doc__
    x_tpos = 0.1  # x position for plotting text of name
    y_tpos = 0.5  # y position for plotting text of name
    return fname, x_tpos, y_tpos



# --------------------------------------------------
if __name__ == '__main__':
    main()
