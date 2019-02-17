from green import *
from dipole import *
from util import *
from metalens import *
import numpy as np

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
starts = np.array([0] * 10)

chi = 4989276.07580808 + 1j * 15803067.7684502

population_size = 20
lens_size = 10


def main(new_focus):
    X = np.array([0])
    Z = np.arange(1000, 10100, 100)
    Y = np.array([0])
    expecting = (0, new_focus)
    population = [gen_random_metalens(lens_size) for _ in range(population_size)]
    print("random population generated")
    it = 0
    while it < 1000 and population[0].score > EPS:
        it += 1
        population = breed(population)
        calc_population_scores(population, expecting, X, Y, Z)
        population.sort()
        population = population[:population_size]
        print("epoch: {}, lowest score: {}".format(it, population[0].score))
    # while it < 1000 and distance(current_focus, expecting) > EPS:
    #     new = mutate(current_subject)
    #     focus = calc(new, X, Y, Z, False)
    #     if distance(focus, expecting) < distance(current_focus, expecting) or \
    #             (distance(focus, expecting) == distance(current_focus, expecting) and simpler(new, current_subject)):
    #         current_focus = focus
    #         current_subject = new
    #     it += 1
    #     print(it, focus, distance(focus, expecting))
    #     print()
    current_subject = population[0]
    draw_points(get_points(current_subject), expecting, it)
    X = np.arange(-2000, 2100, 100)
    Z = np.arange(1000, 10050, 50)
    print("drawing colormap")
    draw_colormap(-2000, 2000, 1000, 10000, calc(current_subject, X, Y, Z, True), expecting)
    return current_subject


def calc_population_scores(population: [Metalens], expecting, X, Y, Z):
    for lens in population:
        lens.score = distance(calc(lens, X, Y, Z, False), expecting)


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
            # dop_m = np.exp(1j * np.dot(w, np.array((X[i], Y[k], Z[j])))) / k0
            # psi_0_m[i, j] = dop_m * np.cross(E_in, w)
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
                # g_m = 1j * k0 * rot_green(X[i], dipole_x, Y[k], dipole_y, Z[j], dipole_z, k1)
                # s_m += np.dot(g_m, dipole.moment)
                s += np.dot(g, dipole.moment)
            psi_2[i, j] = psi_2[i, j] + s
            psi_m[i, j] = psi_m[i, j] + s_m

    intensity = np.zeros((len_X, len_Z))
    for i in range(len_X):
        for j in range(len_Z):
            for t in range(3):
                intensity[i, j] += np.abs(psi_2[i, j, t]) ** 2
                # intensity[i, j] += np.abs(psi_m[i, j, t]) ** 2

    m = np.transpose(intensity)
    max_z, max_x = np.unravel_index(m.argmax(), m.shape)
    if get_intensity:
        return intensity
    else:
        return X[max_x], Z[max_z]


if __name__ == '__main__':
    np.random.seed(322)
    for i in range(5300, 6000, 100):
        print("building lens with focus (0, {})".format(i))
        start_time = time.time()
        main(i)
        print(time.time() - start_time)
