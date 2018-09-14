#!/usr/bin/env python3
"""
Solution for ISTA 421 / INFO 521 Fall 2015, HW 2, Problem 1
Author: Clayton T. Morrison, 12 September 2015
        Ken Youens-Clark (Sept 2018)
"""

import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


# --------------------------------------------------
def get_args():
    """get args"""
    parser = argparse.ArgumentParser(
        description='Find w-hat',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('file', metavar='FILE', help='csv data file')

    parser.add_argument(
        '-m',
        '--model_order',
        help='Model order',
        metavar='int',
        type=int,
        default=1)

    parser.add_argument(
        '-t',
        '--title',
        help='Plot title',
        metavar='str',
        type=str,
        default=None)

    parser.add_argument(
        '-x',
        '--xlabel',
        help='X axis label',
        metavar='str',
        type=str,
        default='x')

    parser.add_argument(
        '-y',
        '--ylabel',
        help='Y axis label',
        metavar='str',
        type=str,
        default='t')

    parser.add_argument(
        '-o',
        '--outfile',
        help='Save output to filename',
        metavar='str',
        type=str,
        default=None)

    parser.add_argument(
        '-s', '--scale', help='Whether to scale the data', action='store_true')

    parser.add_argument(
        '-q',
        '--quiet',
        help='Do not show debug messages',
        action='store_true')

    return parser.parse_args()


# --------------------------------------------------
def die(msg='Something bad happened'):
    """warn() and exit with error"""
    logging.critical(msg)
    sys.exit(1)


# --------------------------------------------------
def main():
    """main"""
    args = get_args()

    logging.basicConfig(
        level=logging.CRITICAL if args.quiet else logging.DEBUG)

    title = args.title if args.title else 'Fit to {} order'.format(
        args.model_order)

    read_data_fit_plot(
        args.file,
        model_order=args.model_order,
        scale_p=args.scale,
        plot_title=title,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        save_path=args.outfile,
        plot_p=True)


# --------------------------------------------------
def plot_data(x, t, title='Data', xlabel='x', ylabel='t'):
    """
    Plot single input feature x data with corresponding response
    values t as a scatter plot.

    :param x: sequence of 1-dimensional input data values (numbers)
    :param t: sequence of 1-dimensional responses
    :param title: title of plot (default 'Data')
    :return: None
    """
    plt.figure()  # Create a new figure object for plotting
    plt.scatter(x, t, edgecolor='b', color='w', marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.pause(.1)  # required on some systems to allow rendering


# --------------------------------------------------
def plot_model(x, w):
    """
    Plot the function (curve) of an n-th order polynomial model:
        t = w0*x^0 + w1*x^1 + w2*x^2 + ... wn*x^n
    This works by creating a set of x-axis (plotx) points and
    then use the model parameters w to determine the corresponding
    t-axis (plott) points on the model curve.

    :param x: sequence of 1-dimensional input data values (numbers)
    :param w: n-dimensional sequence of model parameters: w0, w1, w2, ..., wn
    :return: the plotx and plott values for the plotted curve
    """
    # NOTE: this assumes a figure() object has already been created.

    # plotx represents evenly-spaced set of 100 points on the x-axis
    # used for creating a relatively "smooth" model curve plot.
    # Includes points a little before the min x input (-0.25)
    # and a little after the max x input (+0.25)
    plotx = np.linspace(min(x) - 0.25, max(x) + 0.25, 100)

    # plotX (note that python is case sensitive, so this is *not*
    # the same as plotx with a lower-case x) is the "design matrix"
    # for our model curve inputs represented in plotx.
    # We need to do the same computation as we do when doing
    # model fitting (as in fitpoly(), below), except that we
    # don't need to infer (by the normal equations) the values
    # of w, as they are given here as input.
    # plotx.shape[0] ensures we create a matrix with the number of
    # rows corresponding to the number of points in plotx (this will
    # still work even if we change the number of plotx points to
    # something other than 100)
    plotX = np.zeros((plotx.shape[0], w.size))

    # populate the design matrix plotX
    for k in range(w.size):
        plotX[:, k] = np.power(plotx, k)

    # Take the dot (inner) product of the design matrix plotX and the
    # parameter vector w
    plott = np.dot(plotX, w)

    # plot the x (plotx) and t (plott) values in red
    plt.plot(plotx, plott, color='r', linewidth=2)

    plt.pause(.1)  # required on some systems to allow rendering
    return plotx, plott


# --------------------------------------------------
def scale01(x):
    """
    HELPER FUNCTION: only needed if you are working with large
    x values.  This is NOT needed for problems 1, 2 and 4.

    Mathematically, the sizes of the values of variables (e.g., x)
    could be arbitrarily large.  However, when we go to manipulate
    those values on a computer, we need to be careful.  E.g., in
    these exercises we are taking powers of values and if the
    values are large, then taking a large power of the variable
    may exceed what can be represented numerically.

    For example, in the Olympics data (both men's and women's),
    the input x values are years in the 1000's.  If you model
    is, say, polynomial order 5, then you're taking a large
    number to the power of 5, which is the order of a quadrillion!
    Python floating point numbers have trouble representing this
    many significant digits.

    This function here scales the input data to be the range [0, 1]
    (i.e., between 0 and 1, inclusive).

    :param x: sequence of 1-dimensional input data values (numbers)
    :return: x values linearly scaled to range [0, 1]
    """
    x_min = min(x)
    x_range = max(x) - x_min
    return (x - x_min) / x_range


# --------------------------------------------------
def fitpoly(x, t, model_order):
    """
    Given "training" data in input sequence x (number features),
    corresponding target value sequence t, and a specified
    polynomial of order model_order, determine the linear
    least mean squared (LMS) error best fit for parameters w,
    using the generalized matrix normal equation.

    model_order is a non-negative integer, n, representing the
    highest polynomial exponent of the polynomial model:
        t = w0*x^0 + w1*x^1 + w2*x^2 + ... wn*x^n

    :param x: sequence of 1-dimensional input data features
    :param t: sequence of target response values
    :param model_order: integer representing the highest polynomial exponent of the polynomial model
    :return: parameter vector w
    """

    # Construct the empty design matrix
    # np.zeros takes a python tuple representing the number
    # of elements along each axis and returns an array of those
    # dimensions filled with zeros.
    # For example, to create a 2x3 array of zeros, call
    #     np.zeros((2,3))
    # and this returns (if executed at the command-line):
    #     array([[ 0.,  0.,  0.],
    #            [ 0.,  0.,  0.]])
    # The number of columns is model_order+1 because a model_order
    # of 0 requires one column (filled with input x values to the
    # power of 0), model_order=1 requires two columns (first input x
    # values to power of 0, then column of input x values to power 1),
    # and so on...
    X = np.zeros((x.shape[0], model_order + 1))

    # Fill each column of the design matrix with the corresponding
    for k in range(model_order + 1):  # w.size
        X[:, k] = np.power(x, k)

    logging.debug('model_order = {}'.format(model_order))
    logging.debug('x.shape = {}'.format(x.shape))
    logging.debug('X.shape = {}'.format(X.shape))
    logging.debug('t.shape = {}'.format(t.shape))

    #w, res, rank, s = np.linalg.lstsq(X, t)
    # w = (X^T . X)^-1 * (X^T . t)
    w = np.linalg.inv((X.transpose().dot(X))).dot(X.transpose().dot(t))

    logging.debug('w.shape = {}'.format(w.shape))

    return w


# --------------------------------------------------
def read_data_fit_plot(data_path,
                       model_order=1,
                       scale_p=False,
                       save_path=None,
                       plot_p=False,
                       xlabel='x',
                       ylabel='y',
                       plot_title='Data'):
    """
    A "top-level" script to
        (1) Load the data
        (2) Optionally scale the data between [0, 1]
        (3) Plot the raw data
        (4) Find the best-fit parameters
        (5) Plot the model on top of the data
        (6) If save_path is a filepath (not None), then save the figure as a pdf
        (6) Optionally call the matplotlib show() fn, which keeps the plot open
    :param data_path: Path to the data
    :param model_order: Non-negative integer representing model polynomial order
    :param scale_p: Boolean Flag (default False)
    :param save_path: Optional (default None) filepath to save figure to file
    :param plot_p: Boolean Flag (default False)
    :param plot_title: Title of the plot (default 'Data')
    :return: None
    """

    # (1) load the data
    data = np.genfromtxt(data_path, delimiter=',', dtype=None)

    # (2) Optionally scale the data between [0,1]
    # See the scale01 documentation for explanation of why you
    # might want to scale
    if scale_p:
        x = scale01(
            data[:,
                 0])  # extract x (slice first column) and scale so x \in [0,1]
    else:
        x = data[:, 0]  # extract x (slice first column)

    t = data[:, 1]  # extract t (slice second column)

    # (3) plot the raw data
    plot_data(x, t, title=plot_title, xlabel=xlabel, ylabel=ylabel)

    # (4) find the best-fit model parameters using the fitpoly function
    w = fitpoly(x, t, model_order)

    logging.debug(
        'Identified model parameters w (in scientific notation):\n{}'.format(
            w))

    # python defaults to print floats in scientific notation,
    # so here I'll also print using python format, which I find easier to read
    logging.debug('w again (not in scientific notation):\n{}'.format(
        ['{0:f}'.format(i) for i in w]))

    # (5) Plot the model on top of the data
    plot_model(x, w)

    # (6) If save_path is a filepath (not None), then save the figure as a pdf
    if save_path is not None:
        plt.savefig(save_path, fmt='pdf')

    # (7) Optionally show the plot window (and hold it open)
    if plot_p:
        plt.show()


# --------------------------------------------------
if __name__ == '__main__':
    main()
