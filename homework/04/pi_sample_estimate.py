#!/usr/bin/env python3
"""
Estimate pi by sampling N points in a quarter of a unit circle
and use Pythagorean theorem (a^2 + b^2 = c^2)
Author: Ken Youens-Clark
Date: October 22, 2018
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

    parser.add_argument('-n', '--num_samples',
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
    """main"""
    args = get_args()
    num_samples = args.num_samples

    if num_samples < 1:
        die('-n ({}) cannot be less than 1'.format(num_samples))

    count = 0
    for _ in range(0, num_samples):
        x, y = random(), random()
        if hypot(x, y) <= 1:
            count += 1

    print('pi ≃ {:.06f}'.format(count * 4 / num_samples))

# --------------------------------------------------
if __name__ == '__main__':
    main()
