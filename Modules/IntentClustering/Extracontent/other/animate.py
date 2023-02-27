# @Author: shounak.ray
# @Date:   2022-06-28T19:49:57-07:00
# @Last modified by:   shounak.ray
# @Last modified time: 2022-06-28T21:27:15-07:00


#! /usr/bin/env python3

import itertools

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from matplotlib import colors

np.random.seed(0)

# hyperparameters for SOM
niters = 750
init_learning_rate = 0.1

h, w, n = 50, 50, 3
grid = np.random.rand(h, w, n)
init_map_radius = max(h, w) / 2
time_constant = niters / np.log(init_map_radius)

# assume the data is infinite and all 8 colors are equally probable
data = np.array([
    colors.BASE_COLORS[x] for x in colors.BASE_COLORS.keys()
])

frames = np.zeros((31, h, w, n))
frames[0] += grid


def animate_func(i):
    im.set_array(frames[i])
    return [im]


fig = plt.figure()
im = plt.imshow(frames[0], interpolation='none', aspect='auto', vmin=0, vmax=1)

for t in range(1, niters + 1):
    if t % 25 == 0:
        print(t)
        frames[t] += grid

    datum = data[np.random.choice(
        np.arange(data.shape[0])
    ), :]

    dists = np.square(grid - datum).sum(axis=2)
    dmin_idx = dists.argmin()
    x, y = np.unravel_index(dmin_idx, grid.shape[0:2])
    bmu = grid[x, y]

    radius = init_map_radius * np.exp(-0.5 * t / time_constant)
    lr = init_learning_rate * np.exp(-t / niters)

    for i, j in itertools.product(range(h), range(w)):
        dist_sq = (x - i)**2 + (y - j)**2
        if dist_sq < radius**2:
            theta = np.exp(-0.5 * dist_sq / (radius**2))
            grid[i, j, :] += theta * lr * (datum - grid[i, j, :])


anim = animation.FuncAnimation(
    fig, animate_func, frames=frames.shape[0], interval=100
)
anim.save('test_anim.mp4')
