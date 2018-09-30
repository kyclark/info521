#!/usr/bin/env python3

# Author: Ken Youens-Clark
# INFO521 Homeword 3 Problem 6

import numpy as np
import matplotlib.pyplot as plt


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


# --------------------------------------------------
def main():
    xmin = -4.
    xmax = 5.
    noise_var = 6
    orders = [1, 3, 5, 9]
    N = 25
    num_samples = 20

    # Make a set of N evenly-spaced x values between xmin and xmax
    test_x = np.linspace(xmin, xmax, N)
    true_y = true_function(test_x)

    for i in orders:
        plt.figure(0)

        for _ in range(0, num_samples):
            x, t = sample_from_function(
                N=25, xmin=xmin, xmax=xmax, noise_var=noise_var)
            X = np.zeros(shape=(x.shape[0], i + 1))
            testX = np.zeros(shape=(test_x.shape[0], i + 1))
            for k in range(i + 1):
                X[:, k] = np.power(x, k)
                testX[:, k] = np.power(test_x, k)

            # fit model parameters
            w = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, t))

            # calculate predictions
            prediction_t = np.dot(testX, w)
            plt.plot(test_x, prediction_t, color='blue')

        # Plot the true function in red so it will be visible
        plt.plot(test_x, true_y, color='red', linewidth=3)
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Model order {} prediction of {}, $x \in [{},{}]$'.format(
            i, true_function.__doc__, xmin, xmax))
        plt.pause(.1)  # required on some systems so that rendering can happen
        plt.show()


# --------------------------------------------------
if __name__ == '__main__':
    main()
