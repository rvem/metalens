from green import *
from dipole import *
from util import *
from metalens import *
import numpy as np
import matplotlib.animation as animation
import time

pi = np.pi

EPS = 1
# lambda_ = 826.60
lambda_ = 620
r_sph = 100

eps1 = 1
eps2 = 1
k1 = 2 * pi * eps1 ** (1 / 2) / lambda_
k2 = 2 * pi * eps2 ** (1 / 2) / lambda_
k0 = 2 * pi / lambda_

theta_inc = 0
W_0 = 300000

# polarization vector
E_in = np.array((np.cos(theta_inc), 0, -np.sin(theta_inc)))
# wave vector
w = np.array((k0 * eps1 ** (1 / 2) * np.sin(theta_inc), 0, k0 * eps1 ** (1 / 2) * np.cos(theta_inc)))

rings = np.array([1350, 2350, 4350, 6350, 8350, 9900, 11200, 12450, 13600, 14750])
nums = np.array([8, 8, 16, 32, 32, 32, 64, 64, 64, 128])
starts = np.array([0] * 15)

chi = 4989276.07580808 + 1j * 15803067.7684502

images = []


def main(new_focus, random_seed):
    X = np.array([0])
    Z = np.arange(1000, 10100, 100)
    Y = np.array([0])
    current_subject = gen_random_metalens(10)
    current_subject.it = 0
    current_subject.focus = calc(current_subject, X, Y, Z, False)
    it = 0
    expecting = (0, new_focus)
    trampling_steps = 0
    lowest_score = +np.inf
    images.append(draw_lens(current_subject))
    while it < 5000 and lowest_score > EPS:
        new = mutate(current_subject)
        new_focus = calc(new, X, Y, Z, False)
        new.score = distance(new_focus, expecting)
        new.focus = new_focus
        new.random_seed = random_seed
        if new.score < lowest_score:
            trampling_steps = 0
            lowest_score = new.score
            current_subject = new
            current_subject.focus = new_focus
            current_subject.it = it
            images.append(draw_lens(current_subject))
        else:
            trampling_steps += 1
        if trampling_steps > 100:
            print("generate new random individual after trampling")
            current_subject = gen_random_metalens(10)
            trampling_steps = 0
            lowest_score = distance(calc(current_subject, X, Y, Z, False), expecting)
        it += 1
        if not it % 50:
            print("epoch: {}, lowest score: {}".format(it, lowest_score))
    current_subject.it = it
    draw_lens(current_subject)
    export_as_json(current_subject)

    return current_subject


def draw_heatmap(subject: Metalens):
    X = np.arange(-2000, 2100, 100)
    Z = np.arange(1000, 10050, 50)
    Y = np.array([0])
    print("drawing colormap")
    draw_colormap(-2000, 2000, 1000, 10000, calc(subject, X, Y, Z, True), subject.focus)


def draw_lens(subject: Metalens):
    return draw_points(get_points(subject), subject.focus, subject.it, subject.random_seed, sum(subject.nums))


def calc(ring_subject: Metalens, X, Y, Z, get_intensity: bool):
    len_X = len(X)
    len_Z = len(Z)
    dipoles = [Dipole(x, E_in * chi) for x in get_points(ring_subject)]
    n = len(dipoles)
    psi_0_2 = np.zeros((len_X, len_Z, 3), dtype=complex)
    psi_0_m = np.zeros((len_X, len_Z, 3), dtype=complex)
    k = 0
    for i in range(len_X):
        for j in range(len_Z):
            dop = np.exp(1j * np.dot(w, np.array((X[i], Y[k], Z[j]))))
            psi_0_2[i, j] = dop * E_in
            dop_m = np.exp(1j * np.dot(w, np.array((X[i], Y[k], Z[j])))) / k0
            psi_0_m[i, j] = dop_m * np.cross(E_in, w)
    psi_2 = psi_0_2
    psi_m = psi_0_m
    for i in range(len_X):
        for j in range(len_Z):
            s = np.zeros((1, 3), dtype=complex)
            s_m = np.zeros((1, 3), dtype=complex)
            for t in range(n):
                dipole = dipoles[t]
                dipole_x = dipole.vector[0]
                dipole_y = dipole.vector[1]
                dipole_z = dipole.vector[2]
                g = green(X[i], dipole_x, Y[k], dipole_y, Z[j], dipole_z, k1, eps1)
                g_m = 1j * k0 * rot_green(X[i], dipole_x, Y[k], dipole_y, Z[j], dipole_z, k1)
                s_m += np.dot(g_m, dipole.moment)
                s += np.dot(g, dipole.moment)
            psi_2[i, j] = psi_2[i, j] + s
            psi_m[i, j] = psi_m[i, j] + s_m

    intensity = np.zeros((len_X, len_Z))
    for i in range(len_X):
        for j in range(len_Z):
            for t in range(3):
                intensity[i, j] += np.abs(psi_2[i, j, t]) ** 2
                intensity[i, j] += np.abs(psi_m[i, j, t]) ** 2

    m = np.transpose(intensity)
    max_z, max_x = np.unravel_index(m.argmax(), m.shape)
    if get_intensity:
        return intensity
    else:
        return X[max_x], Z[max_z]


fig = plt.figure()
ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
line, = ax.plot([], [], lw=2)


def animate(ind):
    line.set_data(images[ind])
    return line,


def init():
    line.set_data([], [])
    return line,


if __name__ == '__main__':
    for i in range(6000, 6100, 100):
        # for j in range(322, 333):
        np.random.seed(322)
        print("building lens with focus (0, {})".format(i))
        start_time = time.time()
        main(i, 322)
        print(time.time() - start_time)
        ani = animation.ArtistAnimation(fig, images, interval=500, repeat=False)
        ani.save("lens_evolution.mp4", writer='imagemagick')
