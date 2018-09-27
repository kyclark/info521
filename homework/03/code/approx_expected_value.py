#!/usr/bin/env python3

# approx_expected_value_sin.py
# Port of approx_expected_value.m
# From A First Course in Machine Learning, Chapter 2.
# Simon Rogers, 01/11/11 [simon.rogers@glasgow.ac.uk]
# Approximating expected values via sampling
import numpy
# NOTE: If you are running on MacOSX and encounter an error like the following:
#     RuntimeError: Python is not installed as a framework.
#     The Mac OS X backend will not be able to function correctly
#     if Python is not installed as a framework...
# Then uncomment the following 2 lines:
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Turn off annoying matplotlib warning
# import warnings
# warnings.filterwarnings("ignore", ".*GUI is implemented.*")


# We are trying to estimate the expected value of
# $f(y) = y^2$
##
# ... where
# $p(y)=U(0,1)$
##
# ... which is given by:
# $\int y^2 p(y) dy$
##
# The analytic result is:
# $\frac{1}{3}$ = 0.333...
# (NOTE: this just gives you the analytic result -- you should be able to derive it!)

# First, let's plot the function, shading the area under the curve for x=[0,1]
# The following plot_fn helps us do this.

# Information about font to use when plotting the function
font = {'family' : 'serif',
        'color'  : 'darkred',
        'weight' : 'normal',
        'size'   : 12,
        }


def plot_fn(fn, a, b, fn_name, resolution=100):
    x = numpy.append(numpy.array([a]),
                     numpy.append(numpy.linspace(a, b, resolution), [b]))

    # x = numpy.linspace(x_begin, x_end, resolution)
    y = fn(x)
    y[0] = 0
    y[-1] = 0
    plt.figure()
    plt.fill(x, y, 'b', alpha=0.3)
    # plt.fill_between(x, y, y2=0, color='b', alpha=0.3)
    fname, x_tpos, y_tpos = fn_name()
    plt.text(x_tpos, y_tpos, fname, fontdict=font)
    plt.title('Area under function')
    plt.xlabel('$y$')
    plt.ylabel('$f(y)$')

    x_range = b - a

    plt.xlim(a-(x_range*0.1), b+(x_range*0.1))


# Define the function y that we are going to plot
def y_fn(x):
    return numpy.power(x, 2)


def y_fn_name():
    """
    Helper for displaying the name of the fn in the plot
    Returns the parameters for plotting the y function name,
    used by approx_expected_value
    :return: fname, x_tpos, y_tpos, year
    """
    fname = r'$f(y) = y^2$'  # latex format of fn name
    x_tpos = 0.1  # x position for plotting text of name
    y_tpos = 0.5  # y position for plotting text of name
    return fname, x_tpos, y_tpos


# Plot the function!
plot_fn(y_fn, 0, 1, y_fn_name)


# Now we'll approximate the area under the curve using sampling...

# Sample 1000 uniformly random values in [0..1]
ys = numpy.random.uniform(low=0.0, high=1.0, size=1000)
# compute the expectation of y, where y is the function that squares its input
ey2 = numpy.mean(numpy.power(ys, 2))
print('\nSample-based approximation: {:f}'.format(ey2))

# Store the evolution of the approximation, every 10 samples
sample_sizes = numpy.arange(1, ys.shape[0], 10)
ey2_evol = numpy.zeros((sample_sizes.shape[0]))  # storage for the evolving estimate...
# the following computes the mean of the sequence up to i, as i iterates
# through the sequence, storing the mean in ey2_evol:
for i in range(sample_sizes.shape[0]):
    ey2_evol[i] = numpy.mean(numpy.power(ys[0:sample_sizes[i]], 2))

# Create plot of evolution of the approximation
plt.figure()
# plot the curve of the estimation of the expected value of f(x)=y^2
plt.plot(sample_sizes, ey2_evol)
# The true, analytic result of the expected value of f(y)=y^2 where y ~ U(0,1): $\frac{1}{3}$
# plot the analytic expected result as a red line:
plt.plot(numpy.array([sample_sizes[0], sample_sizes[-1]]),
         numpy.array([1./3, 1./3]), color='r')
plt.xlabel('Sample size')
plt.ylabel('Approximation of expectation')
plt.title('Approximation of expectation of $f(y) = y^2$')
plt.pause(.1)  # required on some systems so that rendering can happen

plt.show()  # keeps the plot open



