import numpy as np

IDENTITY = np.identity(3)


def green(x, x1, y, y1, z, z1, k, eps):
    rx = x - x1
    ry = y - y1
    rz = z - z1
    G = np.zeros((3, 3), dtype=complex)
    if rx == 0 and ry == 0 and rz == 0:
        return G
    else:
        d = (rx ** 2 + ry ** 2 + rz ** 2) ** 0.5
        f1 = np.exp(1j * k * d) / (4 * np.pi * d * eps)
        f2 = (3 - k ** 2 * d ** 2 - 1j * 3 * k * d) / (d ** 4)
        f3 = (1j * k * d - 1) / (d ** 2)
        G[0, 0] = f1 * (k ** 2 + f2 * rx ** 2 + f3)
        G[1, 0] = f1 * f2 * rx * ry
        G[2, 0] = f1 * f2 * rx * rz
        G[0, 1] = G[1, 0]
        G[1, 1] = f1 * (k ** 2 + f2 * ry ** 2 + f3)
        G[2, 1] = f1 * f2 * ry * rz
        G[0, 2] = G[2, 0]
        G[1, 2] = G[2, 1]
        G[2, 2] = f1 * (k ** 2 + f2 * rz ** 2 + f3)
        return G


def rot_green(x, x1, y, y1, z, z1, k):
    rx = x - x1
    ry = y - y1
    rz = z - z1
    G = np.zeros((3, 3), dtype=complex)
    if rx == 0 and ry == 0 and rz == 0:
        return G
    else:
        d = (rx ** 2 + ry ** 2 + rz ** 2) ** 0.5
        f1 = np.exp(1j * k * d) / (4 * np.pi * d)
        f2 = (1 / (d ** 2) - 1j * k / d)
        G[0, 0] = 0
        G[1, 0] = -f1 * f2 * rz
        G[2, 0] = f1 * f2 * ry
        G[0, 1] = - G[1, 0]
        G[1, 1] = 0
        G[2, 1] = -f1 * f2 * rx
        G[0, 2] = -G[2, 0]
        G[1, 2] = -G[2, 1]
        G[2, 2] = 0
        return G
