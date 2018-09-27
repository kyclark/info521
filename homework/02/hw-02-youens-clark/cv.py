#!/usr/bin/env python3

# Script for ISTA 421 / INFO 521 Fall 2016, HW 2, Problem 4
# Author: Clayton T. Morrison, 13 September 2015
#         Ken Youens-Clark
# Updated: 3 September 2018
# Based on cv_demo.m
# From A First Course in Machine Learning, Chapter 1.
# Simon Rogers, 31/10/11 [simon.rogers@glasgow.ac.uk]

import argparse
import logging
import os
import sys
import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------
def get_args():
    """get args"""
    parser = argparse.ArgumentParser(
        description='Run cross-validation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('file', metavar='FILE', help='Data file')

    parser.add_argument(
        '-k', help='Number of cross-folds', metavar='int', type=int, default=5)

    parser.add_argument(
        '-m',
        '--max_order',
        help='Maxium order of model',
        metavar='int',
        type=int,
        default=7)

    parser.add_argument(
        '-o',
        '--out_file',
        help='Filename to save figure',
        metavar='str',
        type=str,
        default=None)

    parser.add_argument(
        '-n',
        '--num_runs',
        help='Number of runs',
        metavar='int',
        type=int,
        default=1)

    return parser.parse_args()


# --------------------------------------------------
def die(msg='Something bad happened'):
    """warn() and exit with error"""
    logging.critical(msg)
    sys.exit(1)


# --------------------------------------------------
# Utilities

# The following permutation tools are used below in run_cv() as a convenience
# for randomly sorting the order of the data (more specifically, the indices
# of the input data)


# --------------------------------------------------
def random_permutation_matrix(n):
    """
    Generate a permutation matrix: an NxN matrix in which each row
    and each column has only one 1, with 0's everywhere else.
    See: https://en.wikipedia.org/wiki/Permutation_matrix
    :param n: size of the square permutation matrix
    :return: NxN permutation matrix
    """
    rows = np.random.permutation(n)
    cols = np.random.permutation(n)
    m = np.zeros((n, n))
    for r, c in zip(rows, cols):
        m[r][c] = 1
    return m


# --------------------------------------------------
def permute_rows(X, P=None):
    """
    Permute the rows of a 2-d array (matrix) according to
    permutation matrix P: left-multiply X by P
    If no P is provided, a random permutation matrix is generated.
    :param X: 2-d array
    :param P: Optional permutation matrix; default=None
    :return: new version of X with rows permuted according to P
    """
    if P is None:
        P = random_permutation_matrix(X.shape[0])
    return np.dot(P, X)


# --------------------------------------------------
def permute_cols(X, P=None):
    """
    Permute the columns of a 2-d array (matrix) according to
    permutation matrix P: right-multiply X by P
    If no P is provided, a random permutation matrix is generated.
    :param X: 2-d array
    :param P: Optional permutation matrix; default=None
    :return: new version of X with columns permuted according to P
    """
    if P is None:
        P = random_permutation_matrix(X.shape[0])
    return np.dot(X, P)


# --------------------------------------------------
# Utilities from fitpoly


# --------------------------------------------------
def read_data(filepath, d=','):
    """ returns an np.array of the data """
    return np.genfromtxt(filepath, delimiter=d, dtype=None)


# --------------------------------------------------
def plot_data(x, t):
    """
    Plot single input feature x data with corresponding response
    values t as a scatter plot
    :param x: sequence of 1-dimensional input data features
    :param t: sequence of 1-dimensional responses
    :return: None
    """
    plt.figure()  # Create a new figure object for plotting
    plt.scatter(x, t, edgecolor='b', color='w', marker='o')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Data')
    plt.pause(.1)  # required on some systems so that rendering can happen


# --------------------------------------------------
def plot_model(x, w, color='r'):
    """
    Plot the curve for an n-th order polynomial model:
        t = w0*x^0 + w1*x^1 + w2*x^2 + ... wn*x^n
    This works by creating a set of x-axis (plotx) points and
    then use the model parameters w to determine the corresponding
    t-axis (plott) points on the model curve.
    :param x: sequence of 1-dimensional input data features
    :param w: n-dimensional sequence of model parameters: w0, w1, w2, ..., wn
    :param color: matplotlib color to plot model curve
    :return: the plotx and plott values for the plotted curve
    """
    # NOTE: this assumes a figure() object has already been created.
    plotx = np.linspace(min(x) - 0.25, max(x) + 0.25, 100)
    plotX = np.zeros((plotx.shape[0], w.size))
    for k in range(w.size):
        plotX[:, k] = np.power(plotx, k)
    plott = np.dot(plotX, w)
    plt.plot(plotx, plott, color=color, linewidth=2)
    plt.pause(.1)  # required on some systems so that rendering can happen
    return plotx, plott


# --------------------------------------------------
# Synthetic data generation


# --------------------------------------------------
def generate_synthetic_data(N, w, xmin=-5, xmax=5, sigma=150):
    """
    Generate some synthetic data
    :param N: Number of sample points
    :param w: np array (1d) representing generating model parameters
    :param xmin: x minimum
    :param xmax: x maximum
    :param sigma: standard deviation
    :return:
    """
    # generate N random input x points between [xmin, xmax]
    x = (xmax - xmin) * np.random.rand(N) + xmin

    # generate response with Gaussian random noise
    X = np.zeros((x.size, w.size))
    for k in range(w.size):
        X[:, k] = np.power(x, k)
    t = np.dot(X, w) + sigma * np.random.randn(x.shape[0])

    return x, t


# --------------------------------------------------
def plot_data_and_model(
        x,
        t,
        w,
        title='Plot of synthetic data; green curve is original generating function',
        filepath=None):
    plot_data(x, t)
    plt.title(title)
    plot_model(x, w, color='g')
    if filepath:
        plt.savefig(filepath, format='pdf')


# --------------------------------------------------
def plot_cv_results(train_loss,
                    cv_loss,
                    ind_loss=None,
                    log_scale_p=False,
                    plot_title='Mean Square Error Loss',
                    filepath=None):
    """
    Helper function to plot the results of cross-validation.
    You can use this when you have all three types of loss, or
    if you just have train and cv loss (in this case, set ind_loss to None)
    :param train_loss:
    :param cv_loss:
    :param ind_loss: If None, only display train_loss and cv_loss
    :param log_scale_p: Whether the plots are in log-scale
    :param plot_title: Title for the plot
    :param filepath: If specified, save pdf file
    :return:
    """

    if ind_loss is not None:
        figw, figh, mydpi = 1200, 420, 96
        plt.figure(figsize=(figw / mydpi, figh / mydpi), dpi=mydpi)
    else:
        figw, figh, mydpi = 800, 420, 96
        plt.figure(figsize=(figw / mydpi, figh / mydpi), dpi=mydpi)

    # plt.figure()
    plt.suptitle(plot_title)

    if log_scale_p:
        ylabel = 'Log MSE Loss'
    else:
        ylabel = 'MSE Loss'

    x = np.arange(0, train_loss.shape[0])

    # put y-axis on same scale for all plots
    if ind_loss is not None:
        min_ylim = min(min(train_loss), min(cv_loss), min(ind_loss))
    else:
        min_ylim = min(min(train_loss), min(cv_loss))
    min_ylim = int(np.floor(min_ylim))
    if ind_loss is not None:
        max_ylim = max(max(train_loss), max(cv_loss), max(ind_loss))
    else:
        max_ylim = max(max(train_loss), max(cv_loss))
    max_ylim = int(np.ceil(max_ylim))

    # Plot training loss
    if ind_loss is not None:
        plt.subplot(131)
    else:
        plt.subplot(121)
    plt.plot(x, train_loss, linewidth=2)
    plt.xlabel('Model Order')
    plt.ylabel(ylabel)
    plt.title('Train Loss')
    plt.pause(.1)  # required on some systems so that rendering can happen
    plt.ylim(min_ylim, max_ylim)

    # Plot CV loss
    if ind_loss is not None:
        plt.subplot(132)
    else:
        plt.subplot(122)
    plt.plot(x, cv_loss, linewidth=2)
    plt.xlabel('Model Order')
    plt.ylabel(ylabel)
    plt.title('CV Loss')
    plt.pause(.1)  # required on some systems so that rendering can happen
    plt.ylim(min_ylim, max_ylim)

    # Plot independent-test loss
    if ind_loss is not None:
        plt.subplot(133)
        plt.plot(x, ind_loss, linewidth=2)
        plt.xlabel('Model Order')
        plt.ylabel(ylabel)
        plt.title('Independent Test Loss')
        plt.pause(.1)  # required on some systems so that rendering can happen
        plt.ylim(min_ylim, max_ylim)

    if ind_loss is not None:
        plt.subplots_adjust(right=0.95, wspace=0.4)
    else:
        plt.subplots_adjust(right=0.95, wspace=0.25, bottom=0.2)

    plt.draw()

    if filepath:
        _, ext = os.path.splitext(filepath)
        out_fmt = ext[1:] if ext else 'pdf'
        plt.savefig(filepath, format=out_fmt)


# --------------------------------------------------
def run_cv(K, maxorder, x, t, randomize_data=False, title='CV', outfile=None):
    """

    :param K: Number of folds
    :param maxorder: Integer representing the highest polynomial order
    :param x: input data (1d observations)
    :param t: target data
    :param randomize_data: Boolean (default False) whether to randomize the order of the data
    :param title: Title for plots of results
    :param outfile: Where to save figure
    :return:
    """

    N = x.shape[0]  # number of data points

    # Use when you want to ensure the order of the data has been
    # randomized before splitting into folds
    # Note that in the simple demo here, the data is already in
    # random order.  However, if you use this function more generally
    # for new data, you may need to ensure you're randomizing the
    # order of the data!
    if randomize_data:
        # use the same permutation P on both x and t, otherwise they'll
        # each be in different orders!
        P = random_permutation_matrix(x.size)
        x = permute_rows(x, P)
        t = permute_rows(t, P)

    # Storage for the design matrix used during training
    # Here we create the design matrix to hold the maximum sized polynomial order
    # When computing smaller model polynomial orders below, we'll just use
    # the first 'k+1' columns for the k-th order polynomial.  This way we don't
    # have to keep creating new design matrix arrays.
    X = np.zeros((x.shape[0], maxorder + 1))

    # Create approximately equal-sized fold indices
    # These correspond to indices in the design matrix (X) rows
    # (where each row represents one training input x)
    fold_indices = list(map(int, np.linspace(0, N, K + 1)))

    # storage for recording loss across model polynomial order
    # rows = fold loss (each row is the loss for one fold)
    # columns = model polynomial order
    cv_loss = np.zeros((K, maxorder + 1))  # cross-validation loss
    train_loss = np.zeros((K, maxorder + 1))  # training loss

    # iterate over model polynomial orders
    for p in range(maxorder + 1):

        # Augment the input data by the polynomial model order
        # E.g., 2nd-order polynomial model takes input x to the 0th, 1st, and 2nd power
        X[:, p] = np.power(x, p)

        # iterate over folds
        for fold in range(K):
            # Partition the data
            # foldX, foldt contains the data for just one fold being held out
            # trainX, traint contains all other data

            foldX = X[fold_indices[fold]:fold_indices[fold + 1], 0:p + 1]
            foldt = t[fold_indices[fold]:fold_indices[fold + 1]]

            # safely copy the training data (so that deleting doesn't
            # remove the original
            trainX = np.copy(X[:, 0:p + 1])

            # remove the fold x from the training set
            trainX = np.delete(
                trainX, np.arange(fold_indices[fold], fold_indices[fold + 1]),
                0)

            # safely copy the training data (so that deleting doesn't
            # remove the original
            traint = np.copy(t)

            # remove the fold t from the training set
            traint = np.delete(
                traint, np.arange(fold_indices[fold], fold_indices[fold + 1]),
                0)

            # find the least mean squares fit to the training data
            w = np.linalg.inv((trainX.transpose().dot(trainX))).dot(
                trainX.transpose().dot(traint))

            # calculate and record the mean squared losses
            train_pred = np.dot(trainX,
                                w)  # model predictions on training data
            train_loss[fold, p] = np.mean(np.power(train_pred - traint, 2))

            fold_pred = np.dot(foldX, w)  # model predictions on held-out fold
            cv_loss[fold, p] = np.mean(np.power(fold_pred - foldt, 2))

    # The loss values can get quite large, so take the log for display purposes

    # Ensure taking log of the mean (not mean of the log!)
    mean_train_loss = np.mean(train_loss, 0)
    mean_cv_loss = np.mean(cv_loss, 0)

    log_mean_train_loss = np.log(mean_train_loss)
    log_mean_cv_loss = np.log(mean_cv_loss)

    print('\n----------------------\nResults for {0}'.format(title))
    print('log_mean_train_loss:\n{0}'.format(log_mean_train_loss))
    print('log_mean_cv_loss:\n{0}'.format(log_mean_cv_loss))

    min_mean_log_cv_loss = min(log_mean_cv_loss)

    # TODO: has to be better way to get the min index...
    best_poly = [
        i for i, j in enumerate(log_mean_cv_loss) if j == min_mean_log_cv_loss
    ][0]

    print('minimum mean_log_cv_loss of {0} for order {1}'.format(
        min_mean_log_cv_loss, best_poly))

    # Plot log scale loss results
    plot_cv_results(
        log_mean_train_loss,
        log_mean_cv_loss,
        None,
        log_scale_p=True,
        plot_title=title,
        filepath=outfile)

    # Uncomment to plot direct-scale mean loss results
    # plot_cv_results(mean_train_loss, mean_cv_loss, mean_ind_loss, log_scale_p=False)

    return best_poly, min_mean_log_cv_loss


# --------------------------------------------------
def plot_data2(x, t, title='Data', xlabel='x', ylabel='t'):
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
    plot_data2(x, t, title=plot_title, xlabel=xlabel, ylabel=ylabel)

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
def main():
    """
    main entry point, get command-line args, run multiple iterations
    of CV and LOOCV, plot final best fit
    """
    args = get_args()
    filename = args.file
    K = args.k
    maxorder = args.max_order

    if not os.path.isfile(filename):
        die('Input "{}" is not a file'.format(filename))

    if maxorder < 1:
        die('--max_order "{}" cannot be less than 1'.format(maxorder))

    if args.num_runs < 1:
        die('--num_runs "{}" cannot be less than 1'.format(args.num_runs))

    if K < 1:
        die("-k cannot be less than 1")

    title = '{0}-fold CV'.format(K)

    data = np.genfromtxt(filename, delimiter=',', dtype=None)
    x = data[:, 0]  # all rows, 1st column
    t = data[:, 1]  # all rows, 2nd column

    #
    # First do N iterations for CV, record how many times each order
    # is reported as the best fit
    #
    cv_best_orders = []
    for n in range(args.num_runs):
        best_poly, min_mean_log_cv_loss = run_cv(
            K,
            maxorder,
            x,
            t,
            randomize_data=True,
            title=title,
            outfile='loss_{}cv_o{}_r{}.png'.format(K, maxorder, n + 1))
        cv_best_orders.append(best_poly)

    #
    # Repeat N iterations for LOOCV, record how many times each order
    # is reported as the best fit
    #
    N = data.shape[0]
    loocv_best_orders = []
    for n in range(args.num_runs):
        best_poly, min_mean_log_cv_loss = run_cv(
            N,
            maxorder,
            x,
            t,
            randomize_data=True,
            title='LOOCV',
            outfile='loss_loocv_o{}_r{}.png'.format(maxorder, n + 1))
        loocv_best_orders.append(best_poly)

    #
    # Return the key of a dict as sort by the values (most frequent)
    #
    best_fit_cv = max(set(cv_best_orders), key=cv_best_orders.count)
    best_fit_loocv = max(set(loocv_best_orders), key=loocv_best_orders.count)
    print('Best order for CV   : {}'.format(best_fit_cv))
    print('Best order for LOOCV: {}'.format(best_fit_loocv))

    read_data_fit_plot(
        filename,
        best_fit_loocv,
        scale_p=False,
        save_path='cv-best-fit.png',
        plot_p=True,
        xlabel='x',
        ylabel='t',
        plot_title='Best fit is model order {}'.format(best_fit_loocv))
    plt.show()


# --------------------------------------------------
if __name__ == '__main__':
    main()
