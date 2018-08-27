#!/usr/bin/env python3
"""
Assignment: INFO521 HW1
Date:       31 Aug 2018
Author:     Ken Youens-Clark
"""

import numpy as np
import matplotlib.pyplot as plt

dat = np.loadtxt("../data/humu.txt")
print('type = {}'.format(type(dat)))
print('size = {}'.format(dat.size))
print('shape = {}'.format(dat.shape))
print('max = {}'.format(dat.max()))  # also np.amax(dat)
print('min = {}'.format(dat.min()))  # also np.amin(dat)

scaled = dat / dat.max()
print('scaled min = {} max = {} shape = {}'.format(scaled.min(),
                                                   scaled.max(),
                                                   scaled.shape))

plt.figure()
plt.imshow(dat)
plt.show()

print(plt.cm.cmapname)

plt.imshow(dat, cmap='gray')
plt.show()

outfile = 'random.png'
for _ in range(0, 2):
    ran = np.random.random(dat.shape)
    plt.imshow(ran)
    plt.show()
    np.savetxt(outfile, ran)

ran1 = np.loadtxt(outfile)
plt.imshow(ran1)
plt.show()

print('Done.')
