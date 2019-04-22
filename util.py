import matplotlib.pyplot as plt
from scipy.special import jv, hankel1
from metalens import *

pi = np.pi


def get_alphas(lambda_, r_sph):
    x = 2 * pi * r_sph / lambda_
    kk = 2 * pi / lambda_
    eps = 15.254 + 1j * 0.172
    m = eps ** (1 / 2)
    n = 1

    psi_n = (pi * x / 2) ** (1 / 2) * jv(n + 1 / 2, x)
    psi_n_dir = (1 / 2) * (pi / (2 * x)) ** (1 / 2) * jv(n + 1 / 2, x) + (pi * x / 2) ** (1 / 2) * (
            jv(n - 1 / 2, x) - ((n + 1 / 2) / x) * jv(n + 1 / 2, x))
    psi_n_m = (pi * m * x / 2) ** (1 / 2) * jv(n + 1 / 2, m * x)
    psi_n_dir_m = (1 / 2) * (pi / (2 * m * x)) ** (1 / 2) * jv(n + 1 / 2, m * x) + (pi * m * x / 2) ** (1 / 2) * (
            jv(n - 1 / 2, m * x) - ((n + 1 / 2) / (m * x)) * jv(n + 1 / 2, m * x))
    xci_n = (pi * x / 2) ** (1 / 2) * hankel1(n + 1 / 2, x)
    xci_n_dir = (1 / 2) * (pi / (2 * x)) ** (1 / 2) * hankel1(n + 1 / 2, x) + (pi * x / 2) ** (1 / 2) * (
            hankel1(n - 1 / 2, x) - ((n + 1 / 2) / x) * hankel1(n + 1 / 2, x))

    a_n = (psi_n * psi_n_dir_m - m * psi_n_m * psi_n_dir) / (xci_n * psi_n_dir_m - m * psi_n_m * xci_n_dir)
    b_n = (m * psi_n * psi_n_dir_m - psi_n_m * psi_n_dir) / (m * xci_n * psi_n_dir_m - psi_n_m * xci_n_dir)

    alpha_e = 1j * pi * 6 * a_n / (kk ** 3)
    alpha_m = 1j * pi * 6 * b_n / (kk ** 3)

    return alpha_e, alpha_m


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


def draw_points(a, focus, it, random_seed, n):
    plt.plot([[x[0]] for x in a], [[x[1]] for x in a], 'ro', markersize=1)
    plt.axis([-21000, 21000, -21000, 21000])
    plt.title('lens, focus: ({}, {}) after {} epochs,\n random_seed {}, number of dipoles {}'
              .format(focus[0], focus[1], it, random_seed, n))
    plt.xlabel('X, nm')
    plt.ylabel('Y, nm')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


def draw_colormap(x_min, x_max, z_min, z_max, intensity, type, focus):
    mx = np.max(intensity)
    mn = np.min(intensity)
    c = plt.imshow(np.transpose(intensity), extent=[x_min, x_max, z_max, z_min], cmap=plt.get_cmap('hot'), vmax=mx,
                   vmin=mn)
    plt.title('energy distribution for {}, '.format(type) + 'lens focus: ({}, {})'.format(focus[0], focus[1]))
    plt.xlabel('X, nm')
    plt.ylabel('Z, nm')
    color_bar = plt.colorbar(c, ticks=[mn, mx])
    color_bar.ax.set_yticklabels([mn, mx])
    plt.show()


def distance(expected, actual):
    return ((expected[2][0] - actual[0]) ** 2 + (expected[2][1] - actual[1]) ** 2) ** (1 / 2)


def draw_loss_plot(plot_info, dipoles_num, focus):
    plt.scatter(*zip(*plot_info))
    plt.plot(*zip(*plot_info), '-o')
    plt.title('fitness function plot for generating lens with focus: {}\n total dipoles num: {}'.format(focus, dipoles_num))
    plt.xlabel('epoch')
    plt.ylabel('score')
    plt.savefig("plot_{}.png".format(focus))
    plt.show()
