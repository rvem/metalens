import matplotlib.pyplot as plt
from metalens import *

pi = np.pi


def get_points(subject: Metalens):
    points = []
    for i in range(len(subject.rads)):
        r = subject.rads[i]
        n = subject.nums[i]
        start = subject.starts[i]
        for j in range(n):
            angle = start + (2 * np.pi) / n * j
            points.append(np.array((r * np.cos(angle), r * np.sin(angle), 0)))
    return np.array(points)


def draw_points(a, focus, it):
    plt.plot([[x[0]] for x in a], [[x[1]] for x in a], 'ro', markersize=1)
    plt.axis([-21000, 21000, -21000, 21000])
    plt.title('lens, focus: ({}, {}) after {} steps'.format(focus[0], focus[1], it))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


def draw_colormap(x_min, x_max, z_min, z_max, intensity, focus):
    mx = np.max(intensity)
    mn = np.min(intensity)
    c = plt.imshow(np.transpose(intensity), extent=[x_min, x_max, z_max, z_min], cmap=plt.get_cmap('hot'), vmax=mx,
                   vmin=mn)
    plt.title('energy distribution ' + 'lens, focus: ({}, {})'.format(focus[0], focus[1]))
    plt.xlabel('X')
    plt.ylabel('Z')
    color_bar = plt.colorbar(c, ticks=[mn, mx])
    color_bar.ax.set_yticklabels([mn, mx])
    plt.show()


def distance(expected, actual):
    return ((expected[0] - actual[0]) ** 2 + (expected[1] - actual[1]) ** 2) ** (1 / 2)
