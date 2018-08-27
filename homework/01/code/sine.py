#!/usr/bin/env python3

"""
sine.py - plot sin
INFO521 HW01.11a
Ken Youens-Clark
27 August 2018
"""

import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------
def main():
    lim = 2
    x = np.arange(0, 10, .01)
    y = np.sin(2 * np.pi * x)
    plt.plot(x, y)
    plt.show()
    plt.savefig('sine.png')

if __name__ == '__main__':
    main()
