#!/usr/bin/env python3
#
# INFO521 Homework 3 Problem #1 (Poisson pmf)
# Author: Ken Youens-Clark
# Sept 29, 2018
#

import numpy as np

def p(y, lam):
    return (np.power(lam, y) / np.math.factorial(y)) * np.exp(-1 * lam)

def main():
    lam = 7
    tot = 0
    for y in range(5,11):
        r = p(y, lam)
        print('y({}, {}) = {}'.format(y, lam, r))
        tot += r

    print('total = {}'.format(tot))


if __name__ == '__main__':
    main()
