#!/usr/bin/env python3

# Extension of predictive_variance_example.py
# Port of predictive_variance_example.m
# From A First Course in Machine Learning, Chapter 2.
# Simon Rogers, 01/11/11 [simon.rogers@glasgow.ac.uk]
# Predictive variance example
# Ken Youens-Clark 5 Oct 2018

import matplotlib.pyplot as plt
import numpy as np
import os

# --------------------------------------------------------------------------
# Globals: edit these as needed
# --------------------------------------------------------------------------

# set to True in order to automatically save the generated plots
SAVE_FIGURES = True

# change this to where you'd like the figures saved
# (relative to your python current working directory)
FIGURE_PATH = '.'


# --------------------------------------------------
def calculate_prediction_variance(x_new, X, w, t):
    """
    Calculates the variance for the prediction at x_new
    :param X: Design matrix: matrix of N observations
    :param w: vector of parameters
    :param t: vector of N target responses
    :param x_new: new observation
    :return: the predictive variance around x_new
    """
    s2h = sigma2_hat(X, t, w)
    XTX_inv = np.linalg.inv(np.dot(X.T, X))

    return s2h * np.dot(x_new.T, np.dot(XTX_inv, x_new))


# --------------------------------------------------
def sigma2_hat(X, t, w):
    """
    Calculates sigma-squared-hat from
    :param X: Design matrix: matrix of N observations
    :param t: vector of N target responses
    :param w: vector of parameters
    """
    N = X.shape[0]
    return (1 / N) * (np.dot(t.T, t) - np.dot(np.dot(t.T, X), w))


# --------------------------------------------------
def calculate_cov_w(X, w, t):
    """
    Calculates the covariance of w
    :param X: Design matrix: matrix of N observations
    :param w: vector of parameters
    :param t: vector of N target responses
    :return: the matrix covariance of w
    """
    return sigma2_hat(X, t, w) * np.linalg.inv(np.dot(X.T, X))


# --------------------------------------------------------------------------
# Part 5a -- Generate and Plot the data
# There is nothing you need to implement here
# --------------------------------------------------------------------------


# --------------------------------------------------
def true_function(x):
    """$t = 5x+x^2-0.5x^3$"""
    return (5 * x) + x**2 - (0.5 * x**3)


# --------------------------------------------------
def sample_from_function(N=100, noise_var=1000, xmin=-5., xmax=5.):
    """ Sample data from the true function.
        N: Number of samples
        Returns a noisy sample t_sample from the function
        and the true function t. """
    x = np.random.uniform(xmin, xmax, N)
    t = true_function(x)
    # add standard normal noise using np.random.randn
    # (standard normal is a Gaussian N(0, 1.0)  (i.e., mean 0, variance 1),
    #  so multiplying by np.sqrt(noise_var) make it N(0,standard_deviation))
    t = t + np.random.randn(x.shape[0]) * np.sqrt(noise_var)
    return x, t


xmin = -4.
xmax = 5.
noise_var = 6

# sample 100 points from function
x, t = sample_from_function(100, noise_var, xmin, xmax)

# Chop out some x data:
xmin_remove = -2
xmax_remove = 2

# the following line expresses a boolean function over the values in x;
# this produces a list of the indices of list x for which the test
# was not met; these indices are then deleted from x and t.

pos = ((x >= xmin_remove) & (x <= xmax_remove)).nonzero()
x = np.delete(x, pos, 0)
t = np.delete(t, pos, 0)

# Plot just the sampled data
plt.figure(0)
plt.scatter(np.asarray(x), np.asarray(t), color='k', edgecolor='k')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Sampled data from {0}, $x \in [{1},{2}]$'.format(
    true_function.__doc__, xmin, xmax))
plt.pause(.1)  # required on some systems so that rendering can happen

if SAVE_FIGURES:
    plt.savefig(os.path.join(FIGURE_PATH, 'data.png'), fmt='png')

# Fit models of various orders
orders = [1, 3, 5, 9]

# Make a set of 100 evenly-spaced x values between xmin and xmax
testx = np.linspace(xmin, xmax, 100)

# --------------------------------------------------------------------------
# Part 5b -- Error bar plots
# There is nothing you need to implement here;
# However, you need to complete the implementation of function
# calculate_prediction_variance, above
# --------------------------------------------------------------------------

# Generate plots of predicted variance (error bars) for various model orders
for i in orders:
    # create input representation for given model polynomial order
    X = np.zeros(shape=(x.shape[0], i + 1))
    testX = np.zeros(shape=(testx.shape[0], i + 1))
    for k in range(i + 1):
        X[:, k] = np.power(x, k)
        testX[:, k] = np.power(testx, k)

    # fit model parameters
    w = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, t))

    # calculate predictions
    prediction_t = np.dot(testX, w)

    # get variance, each x_new at a time
    prediction_t_variance = np.zeros(testX.shape[0])
    for j in range(testX.shape[0]):
        x_new = testX[j, :]  # get the x_new observation from row i of testX
        var = calculate_prediction_variance(x_new, X, w,
                                            t)  # calculate prediction variance
        prediction_t_variance[j] = var

    # Plot the data and predictions
    plt.figure()
    plt.scatter(x, t, color='k', edgecolor='k')
    plt.xlabel('x')
    plt.ylabel('t')

    # TODO: fix!
    plt.errorbar(testx, prediction_t, prediction_t_variance)

    # find ylim plot bounds automagically...
    min_model = min(prediction_t - prediction_t_variance)
    max_model = max(prediction_t + prediction_t_variance)
    min_testvar = min(min(t), min_model)
    max_testvar = max(max(t), max_model)
    plt.ylim(min_testvar, max_testvar)

    ti = 'Plot of predicted variance for model with polynomial order {:g}'.format(
        i)
    plt.title(ti)
    plt.pause(.1)  # required on some systems so that rendering can happen

    if SAVE_FIGURES:
        filename = 'error-{0}.png'.format(i)
        plt.savefig(os.path.join(FIGURE_PATH, filename), fmt='png')

# --------------------------------------------------------------------------
# Part 5c -- Plot functions by sampling from cov(\hat{w})
# There is nothing you need to implement here.
# However, you need to complete the implementation of function
# calculate_cov_w, above
# --------------------------------------------------------------------------

# Generate plots of functions whose parameters are sampled based on cov(\hat{w})
num_function_samples = 20
for i in orders:
    # create input representation for given model polynomial order
    X = np.zeros(shape=(x.shape[0], i + 1))
    testX = np.zeros(shape=(testx.shape[0], i + 1))
    for k in range(i + 1):
        X[:, k] = np.power(x, k)
        testX[:, k] = np.power(testx, k)

    # fit model parameters
    w = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, t))

    # Sample functions with parameters w sampled from a Gaussian with
    # $\mu = \hat{\mathbf{w}}$
    # $\Sigma = cov(w)$

    # determine cov(w)
    covw = calculate_cov_w(X, w, t)  # calculate the covariance of w

    # The following samples num_function_samples of w from
    # multivariate Gaussian (normal) with covaraince covw
    wsamp = np.random.multivariate_normal(w, covw, num_function_samples)

    # Calculate means for each function
    prediction_t = np.dot(testX, wsamp.T)

    # Plot the data and functions
    plt.figure()
    plt.scatter(x, t, color='k', edgecolor='k')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.plot(testx, prediction_t, color='b')

    # find reasonable ylim bounds
    plt.xlim(xmin_remove - 2, xmax_remove + 2)  # (-2,4) # (-3, 3)
    min_model = min(prediction_t.flatten())
    max_model = max(prediction_t.flatten())
    min_testvar = min(min(t), min_model)
    max_testvar = max(max(t), max_model)
    plt.ylim(min_testvar, max_testvar)  # (-400,400)

    ti = 'Plot of {0} functions where parameters '\
         .format(num_function_samples, i) + \
         r'$\widehat{\bf w}$ were sampled from' + '\n' + r'cov($\bf w$)' + \
         ' of model with polynomial order {1}' \
         .format(num_function_samples, i)
    plt.title(ti)
    plt.pause(.1)  # required on some systems so that rendering can happen

    if SAVE_FIGURES:
        filename = 'sampled-fns-{0}.png'.format(i)
        plt.savefig(os.path.join(FIGURE_PATH, filename), fmt='png')

plt.show()
