#!/usr/bin/env python3
#
# INFO521 Homework 3 Problem #1 (Poisson pmf)
# Author: Ken Youens-Clark
# Sept 29, 2018
#

import numpy as np

lam = 7
tot = 0
for y in range(5,11):
    r = (np.power(lam, y) / np.math.factorial(y)) * np.exp(-1 * lam)
    print('y({:2}, {}) = {:.02f}'.format(y, lam, r))
    tot += r

print('--------   ----'.format(tot))
print('total    = {:.02f}'.format(tot))
