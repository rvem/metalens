from green import *
from dipole import *
from util import *
from metalens import *
import numpy as np

import time

pi = np.pi

EPS = 1
# lambda_ = 826.60
# lambda_ = 729
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

chi, chi_m = get_alphas(lambda_, r_sph)
# chi = 4989276.07580808 + 1j * 15803067.7684502

population_size = 20
lens_size = 10

elitary = 4
breeded = 12
mutated = 4
total_calc_number = 0


def main(new_focus, random_seed):
    global total_calc_number
    total_calc_number = 0
    plot_info = []
    X = np.array([0])
    Z = np.arange(1000, 10100, 100)
    Y = np.array([0])
    expecting = (0, new_focus)
    population = [gen_random_metalens(lens_size) for _ in range(population_size)]
    print("random population generated")
    it = 0
    trampling_steps = 0
    lowest_score = +np.inf
    while it < 1000 and population[0].score > EPS:
        it += 1
        calc_population_scores(population, expecting, X, Y, Z)
        population.sort()
        if population[0].score >= lowest_score:
            trampling_steps += 1
        else:
            trampling_steps = 0
            lowest_score = population[0].score
        print("epoch: {}, lowest score: {}".format(it, lowest_score))
        plot_info.append((it, lowest_score))
        if trampling_steps > 20:
            print("generate new random population after trampling")
            new_population = [gen_random_metalens(lens_size) for _ in range(population_size)]
            it = 0
            trampling_steps = 0
        else:
            new_population = population[:elitary] + breed_n(population, breeded) + mutate_n(population, mutated)
        population = new_population
    current_subject = population[0]
    current_subject.focus = expecting
    current_subject.random_seed = random_seed
    current_subject.score = 0
    current_subject.it = it
    export_as_json(current_subject)
    draw_points(get_points(current_subject), expecting, it, random_seed, sum(current_subject.nums))
    X = np.arange(-2000, 2100, 100)
    Z = np.arange(1000, 10100, 100)
    draw_loss_plot(plot_info, total_calc_number, expecting)
    print("drawing colormap")
    intensity_e, intensity_m, intensity = calc(current_subject, X, Y, Z, True)
    draw_colormap(-2, 2, 1, 10, intensity_e, "electric", expecting, random_seed)
    draw_colormap(-2, 2, 1, 10, intensity_m, "magnetic", expecting, random_seed)
    draw_colormap(-2, 2, 1, 10, intensity, "summary", expecting, random_seed)
    return current_subject


def calc_population_scores(population: [Metalens], expecting, X, Y, Z):
    for lens in population:
        lens.score = distance(calc(lens, X, Y, Z, False), expecting)


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
    # electric_d, magnetic_d = np.zeros((len_X, len_Z, 3)), np.zeros((len_X, len_Z, 3))
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


def kek():
    rings = np.array([3000])
    nums = np.array([24])
    # nums = np.array([0])
    starts = np.array([0])
    X = np.array([0])
    Z = np.arange(1000, 10100, 100)
    Y = np.array([0])
    lens = Metalens(rings, starts, nums)
    lens.focus_e, lens.focus_m, lens.focus = calc(lens, X, Y, Z, False)
    lens.score = 0
    # draw_points(get_points(lens), lens.focus, 0, 0, sum(lens.nums))
    X = np.arange(-2000, 2100, 100)
    Z = np.arange(1000, 10100, 100)
    intensity_e, intensity_m, intensity = calc(lens, X, Y, Z, True)
    draw_colormap(-2, 2, 1, 10, intensity_e, "electric", lens.focus_e, 0)
    draw_colormap(-2, 2, 1, 10, intensity_m, "magnetic", lens.focus_m, 0)
    draw_colormap(-2, 2, 1, 10, intensity, "summary", lens.focus, 0)


def draw_heatmaps(lens: Metalens):
    X = np.arange(-2000, 2100, 100)
    Z = np.arange(1000, 10100, 100)
    Y = np.array([0])
    intensity_e, intensity_m, intensity = calc(lens, X, Y, Z, True)
    draw_colormap(-2, 2, 1, 10, intensity_e, "electric", lens.focus, lens.random_seed)
    draw_colormap(-2, 2, 1, 10, intensity_m, "magnetic", lens.focus, lens.random_seed)
    draw_colormap(-2, 2, 1, 10, intensity, "summary", lens.focus, lens.random_seed)


if __name__ == '__main__':
    # kek()
    # np.random.seed(322)
    for i in range(325, 333, 1):
        np.random.seed(i)
        print("building lens with focus (0, {})".format(i))
        start_time = time.time()
        main(6000, i)
        print(time.time() - start_time)
