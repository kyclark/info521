#!/usr/bin/env python3
"""
Author:  Ken Youens-Clark
Date:    October 22, 2018
Purpose:

A circle of radius R is inscribed in a square with sides 2R. The area
of the circle is pi*R^2 and the area of the square is (2R)^2 or 4R^2.
Therefore the ratio of the areas is pi/4.

To estimate pi, we will choose N samples from a uniform distribution
between 0 and the radius of the circle (1). We can then use the
Pythagorean theorem (a^2 + b^2 = c^2) (via the math.hypot function
that computes Euclidean distance) to find the distance from the origin
(0,0). If this is less than the radius squared (which is just one
here), then the point falls within the circle.  Because we are only
looking at points in the upper-right quadrant of the circle, multiple
the number found to be within the circle by 4 and then divide by the
number of samples to estimate pi.

Cf. https://en.wikipedia.org/wiki/Approximations_of_%CF%80
"""

import argparse
import sys
from random import random
from math import hypot


# --------------------------------------------------
def get_args():
    """get args"""
    parser = argparse.ArgumentParser(
        description='Estimate pi',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '-n',
        '--num_samples',
        help='Number of samples',
        metavar='int',
        type=int,
        default='1000000')

    return parser.parse_args()


# --------------------------------------------------
def warn(msg):
    """Print a message to STDERR"""
    print(msg, file=sys.stderr)


# --------------------------------------------------
def die(msg='Something bad happened'):
    """warn() and exit with error"""
    warn(msg)
    sys.exit(1)


# --------------------------------------------------
def main():
    """Make a jazz noise here"""
    args = get_args()
    num_samples = args.num_samples

    if num_samples < 1:
        die('-n ({}) cannot be less than 1'.format(num_samples))

    count = 0
    for _ in range(0, num_samples):
        x, y = random(), random()
        if hypot(x, y) <= 1:
            count += 1

    print('pi ~ {:.06f}'.format(count * 4 / num_samples))


# --------------------------------------------------
if __name__ == '__main__':
    main()
