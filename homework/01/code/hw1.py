#!/usr/bin/env python3
"""
Assignment: INFO521 HW1
Date:       31 Aug 2018
Author:     Ken Youens-Clark
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
def main():
    """
    main
    :param: none
    :return: void
    """
    exercise6("humu.txt", "out.txt")
    #exercise9()
    #exercise10c()
    #exercise11()


# --------------------------------------------------
def scale01(arr):
    """
    Linearly scale the values of an array in the range [0,1]
    :param arr: input ndarray
    :return: scaled ndarray
    """
    return arr / arr.max()


# --------------------------------------------------
def exercise6(infile, outfile):
    """
    Read a file into a Numpy ndarray
    :param infile: the file to read
    :param outfile: where to write the output
    :return: void
    """
    dat = np.loadtxt(infile)
    print('type = {}'.format(type(dat)))
    print('size = {}'.format(dat.size))
    print('shape = {}'.format(dat.shape))
    print('max = {}'.format(dat.max()))  # also np.amax(dat)
    print('min = {}'.format(dat.min()))  # also np.amin(dat)

    scaled = scale01(dat)
    print('scaled min = {} max = {} shape = {}'.format(scaled.min(),
                                                       scaled.max(),
                                                       scaled.shape))

    plt.figure()
    plt.imshow(dat)
    plt.savefig('humu-color.png')
    plt.show()

    print(plt.cm.cmapname)

    plt.savefig('humu-gray.png')
    plt.imshow(dat, cmap='gray')
    plt.show()

    for _ in range(0, 2):
        ran = np.random.random(dat.shape)
        plt.imshow(ran)
        plt.show()
        np.savetxt(outfile, ran)

    ran1 = np.loadtxt(outfile)
    plt.imshow(ran1)
    plt.show()

    print('Done.')


# --------------------------------------------------
def exercise9():
    """
    Estimate the randomness of throwing double-sixes
    :param: none
    :return: void
    """

    np.random.seed(seed=8)

    throws = 1000
    dbl6 = 0
    for _ in range(0, throws):
        (n1, n2) = np.random.randint(low=1, high=7, size=2, dtype=int)
        #print('{} {}'.format(n1, n2))
        if n1 == 6 and n2 == 6:
            dbl6 += 1

    print('Threw double-six {:.2}%'.format((dbl6 / throws) * 100))


# --------------------------------------------------
def exercise10a():
    """
    Print two three-dimensional column vectors
    :param: none
    :return: void
    """

    np.random.seed(seed=5)
    a = np.random.rand(3, 1)
    b = np.random.rand(3, 1)
    print(a)
    print(b)


# --------------------------------------------------
def exercise10b():
    """
    Print two three-dimensional column vectors
    :param: none
    :return: void
    """

    np.random.seed(seed=5)
    a = np.random.rand(3, 1)
    b = np.random.rand(3, 1)
    print("a\n {}".format(a))
    print("b\n{}".format(b))
    print("a + b\n{}".format(a + b))
    print("a * b\n{}".format(a * b))
    print("a ・ b\n{}".format(a.transpose().dot(b)))


# --------------------------------------------------
def exercise10c():
    """
    Vector/matrix manipulation
    :param: none
    :return: void
    """

    np.random.seed(seed=2)
    X = np.asmatrix(np.random.rand(3, 3))
    a = np.random.rand(3, 1)
    b = np.random.rand(3, 1)
    print("a\n {}".format(a))
    print("b\n{}".format(b))
    print("X\n{}".format(X))
    print("aTX\n{}".format(a.transpose() * X))
    print("aTXb\n{}".format(a.transpose() * X * b))
    print("X-1\n{}".format(X.getI()))


# --------------------------------------------------
def exercise11():
    """
    Plotting
    :param: none
    :return: void
    """

    x = np.arange(0, 10, .01)
    y = np.sin(2 * np.pi * x)
    plt.plot(x, y)
    plt.title('Sine Function for x from 0.0 to 10.0')
    plt.xlabel('x values')
    plt.ylabel('sin(x)')
    plt.savefig('sine.png')
    plt.show()


# --------------------------------------------------
if __name__ == '__main__':
    main()
