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
H_in = np.array((0, 1, 0))
# wave vector
w = np.array((k0 * eps1 ** (1 / 2) * np.sin(theta_inc), 0, k0 * eps1 ** (1 / 2) * np.cos(theta_inc)))

rings = np.array([1350, 2350, 4350, 6350, 8350, 9900, 11200, 12450, 13600, 14750])
nums = np.array([8, 8, 16, 32, 32, 32, 64, 64, 64, 128])
starts = np.array([0] * 15)

chi, chi_m = get_alphas(lambda_, r_sph)

images = []
total_calc_number = 0

def main(new_focus, random_seed):
    global total_calc_number
    total_calc_number = 0
    plot_info = []
    X = np.array([0])
    Z = np.arange(1000, 10100, 100)
    Y = np.array([0])
    current_subject = empty_metalens(10)
    current_subject.it = 0
    current_subject.focus = calc(current_subject, X, Y, Z, False)
    it = 0
    expecting = (0, new_focus)
    trampling_steps = 0
    lowest_score = +np.inf
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
        else:
            trampling_steps += 1
        if trampling_steps > 50:
            print("generate new random individual after trampling")
            current_subject = empty_metalens(10)
            trampling_steps = 0
            lowest_score = distance(calc(current_subject, X, Y, Z, False), expecting)
        it += 1
        plot_info.append((it, lowest_score))
        # if not it % 50:
        print("epoch: {}, lowest score: {}".format(it, lowest_score))
    current_subject.it = it
    current_subject.focus = expecting
    draw_lens(current_subject)
    draw_loss_plot(plot_info, total_calc_number, expecting)
    export_as_json(current_subject)
    X = np.arange(-2000, 2100, 100)
    Z = np.arange(1000, 10100, 100)
    Y = np.array([0])
    intensity_e, intensity_m, intensity = calc(current_subject, X, Y, Z, True)
    draw_colormap(-2, 2, 1, 10, intensity_e, "electric", expecting, random_seed)
    draw_colormap(-2, 2, 1, 10, intensity_m, "magnetic", expecting, random_seed)
    draw_colormap(-2, 2, 1, 10, intensity, "summary", expecting, random_seed)

    return current_subject

def draw_lens(subject: Metalens):
    return draw_points(get_points(subject), subject.focus, subject.it, subject.random_seed, sum(subject.nums))


def calc_initial_fields(X, Z):
    len_X = len(X)
    len_Z = len(Z)
    electric_initial = np.zeros((len_X, len_Z, 3), dtype=complex)
    magnetic_initial = np.zeros((len_X, len_Z, 3), dtype=complex)
    for i in range(len_X):
        for j in range(len_Z):
            dop = np.exp(1j * np.dot(w, np.array((X[i], 0, Z[j]))))
            electric_initial[i, j] = dop * E_in
            dop_m = np.exp(1j * np.dot(w, np.array((X[i], 0, Z[j])))) / k0
            magnetic_initial[i, j] = dop_m * np.cross(w, E_in)
    return electric_initial, magnetic_initial


def calc_dipoles_fields(particles, X, Z):
    len_X = len(X)
    len_Z = len(Z)
    n = len(particles)
    electric = np.zeros((len_X, len_Z, 3), dtype=complex)
    magnetic = np.zeros((len_X, len_Z, 3), dtype=complex)
    for i in range(len_X):
        for j in range(len_Z):
            e = np.zeros((1, 3), dtype=complex)
            m = np.zeros((1, 3), dtype=complex)
            for t in range(n):
                dipole = particles[t]
                dipole_x = dipole.vector[0]
                dipole_y = dipole.vector[1]
                dipole_z = dipole.vector[2]
                # electric dipoles contribution
                g = green(X[i], dipole_x, 0, dipole_y, Z[j], dipole_z, k1, eps1)
                g_m = -1j * k0 * rot_green(X[i], dipole_x, 0, dipole_y, Z[j], dipole_z, k1)
                m += np.dot(g_m, dipole.electric)
                e += np.dot(g, dipole.electric)
                # magnetic dipoles contribution
                g = 1j * k0 * rot_green(X[i], dipole_x, 0, dipole_y, Z[j], dipole_z, k1)
                g_m = green(X[i], dipole_x, 0, dipole_y, Z[j], dipole_z, k1, eps1)
                e += np.dot(g, dipole.magnetic)
                m += np.dot(g_m, dipole.magnetic)
            electric[i, j] = electric[i, j] + e
            magnetic[i, j] = magnetic[i, j] + m
    return electric, magnetic


def calc_intensities(X, Z, electric, magnetic):
    len_X = len(X)
    len_Z = len(Z)
    intensity_e = np.zeros((len_X, len_Z))
    intensity_m = np.zeros((len_X, len_Z))
    for i in range(len_X):
        for j in range(len_Z):
            for t in range(3):
                intensity_e[i, j] += np.abs(electric[i, j, t]) ** 2
                intensity_m[i, j] += np.abs(magnetic[i, j, t]) ** 2
    return intensity_e, intensity_m


def calc(ring_subject: Metalens, X, Y, Z, get_intensity: bool):
    global total_calc_number
    len_X = len(X)
    len_Z = len(Z)
    total_calc_number += sum(ring_subject.nums)
    dipoles = [Dipole(x, E_in * chi, H_in * chi_m) for x in get_points(ring_subject)]
    electric, magnetic = calc_initial_fields(X, Z)
    electric_d, magnetic_d = calc_dipoles_fields(dipoles, X, Z)
    electric += electric_d
    magnetic += magnetic_d
    intensity_e = np.zeros((len_X, len_Z))
    intensity_m = np.zeros((len_X, len_Z))
    for i in range(len_X):
        for j in range(len_Z):
            for t in range(3):
                intensity_e[i, j] += np.abs(electric[i, j, t]) ** 2
                intensity_m[i, j] += np.abs(magnetic[i, j, t]) ** 2
    intensity = intensity_e + intensity_m
    m = np.transpose(intensity)
    m_e = np.transpose(intensity_e)
    m_m = np.transpose(intensity_m)
    max_z, max_x = np.unravel_index(m.argmax(), m.shape)
    max_z_e, max_x_e = np.unravel_index(m_e.argmax(), m_e.shape)
    max_z_m, max_x_m = np.unravel_index(m_m.argmax(), m_m.shape)
    if get_intensity:
        return intensity_e, intensity_m, intensity
    else:
        return (X[max_x_e], Z[max_z_e]), (X[max_x_m], Z[max_z_m]), (X[max_x], Z[max_z])

# fig = plt.figure()
# ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
# line, = ax.plot([], [], lw=2)


# def animate(ind):
#     line.set_data(images[ind])
#     return line,
#
#
# def init():
#     line.set_data([], [])
#     return line,


if __name__ == '__main__':
    for i in range(322, 333, 1):
        np.random.seed(i)
        # for j in range(322, 333):
        print("building lens with focus (0, {})".format(i))
        start_time = time.time()
        main(6000, i)
        print(time.time() - start_time)
        # ani = animation.ArtistAnimation(fig, images, interval=500, repeat=False)
        # ani.save("lens_evolution.mp4", writer='imagemagick')
