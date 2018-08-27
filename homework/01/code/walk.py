#!/usr/bin/env python3

"""
walk.py - manipulate and plot "walk.txt"
Ken Youens-Clark
27 August 2018
"""

import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
def main():
    """
    main()
    """
    dat = np.loadtxt('../data/walk.txt')
    print('Data  : min "{:5}, max "{:5}", shape "{}"'.format(
        dat.min(), dat.max(), dat.shape))

    abs_min = np.abs(dat.min())
    scaled = (dat + abs_min) / (dat.max() + abs_min)
    print('Scaled: min "{:5}, max "{:5}", shape "{}"'.format(
        scaled.min(), scaled.max(), scaled.shape))

    outfile = '../data/walk_scale01.txt'
    np.savetxt(outfile, scaled)
    print('Scaled data saved to "{}"'.format(outfile))

    plot_1d_array(arr=dat, title='Original', outfile='walk.png')
    plot_1d_array(arr=scaled, title='Scaled', outfile='walk_scaled.png')


# --------------------------------------------------
def plot_1d_array(arr, title=None, outfile=None):
    """
    Plot a 1D array
    :param: arr - a 1D Numpy array
    :param: title - figure title (str)
    :param: outfile - path to write image (str)

    :return: void
    """

    plt.figure()
    if title:
        plt.title(title)
    plt.plot(arr)

    if outfile:
        plt.savefig(outfile)
        print('Wrote to "{}"'.format(outfile))

    plt.show()


# --------------------------------------------------
if __name__ == '__main__':
    main()
