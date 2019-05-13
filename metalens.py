import numpy as np
import jsonpickle
from copy import copy

# mutations = ['radius', 'angle', 'number', 'add', 'remove']
mutations = ['radius', 'angle', 'number']
true_or_false = [True, False]

MAX_NUM = 50
MIN_RAD = 1000
MAX_RAD = 15000
MIN_DIST = 300


class Metalens:
    def __init__(self, rads, starts, nums):
        self.rads = rads
        self.starts = starts
        self.nums = nums
        self.score = np.inf
        self.focus = (-1, -1)
        self.focus_e = (-1, -1)
        self.focus_m = (-1, -1)
        self.random_seed = -1
        self.it = -1

    def __copy__(self):
        return Metalens(copy(self.rads), copy(self.starts), copy(self.nums))

    def __lt__(self, other):
        return self.score < other.score

    def __len__(self):
        return len(self.rads)


def empty_metalens(n):
    rads = np.array([np.random.randint(MIN_RAD, MAX_RAD) for _ in range(n)])
    for i in range(n):
        new_rad = np.random.randint(MIN_RAD, MAX_RAD)
        while get_dist_to_closest_ring(new_rad, rads, i) < MIN_DIST:
            new_rad = np.random.randint(MIN_RAD, MAX_RAD)
        rads[i] = new_rad
    return Metalens(rads, np.array([0] * n), np.array([0] * n))


def gen_random_metalens(n):
    rads = np.array([np.random.randint(MIN_RAD, MAX_RAD) for _ in range(n)])
    for i in range(n):
        new_rad = np.random.randint(MIN_RAD, MAX_RAD)
        while get_dist_to_closest_ring(new_rad, rads, i) < MIN_DIST:
            new_rad = np.random.randint(MIN_RAD, MAX_RAD)
        rads[i] = new_rad
    starts = np.array([np.random.uniform(0, 2 * np.pi) for _ in range(n)])
    nums = np.array([np.random.randint(0, MAX_NUM) for _ in range(n)])
    for i in range(n):
        new_num = np.random.randint(0, MAX_NUM)
        while get_dist(new_num, rads[i]) < MIN_DIST:
            new_num = np.random.randint(0, MAX_NUM)
        nums[i] = new_num
    return Metalens(rads, starts, nums)


def is_alive(lens: Metalens):
    n = len(lens)
    if min(lens.rads) < MIN_RAD or max(lens.rads) > MAX_RAD:
        return False
    for i in range(n):
        if get_dist_to_closest_ring(lens.rads[i], lens.rads, i) < MIN_DIST:
            return False
        if get_dist(lens.nums[i], lens.rads[i]) < MIN_DIST:
            return False
    if sum(lens.nums) > MAX_NUM * n:
        return False
    return True


def get_dist_to_closest_ring(new_rad, rads, index):
    mn = np.inf
    for i in range(len(rads)):
        if abs(rads[i] - new_rad) < mn and i != index:
            mn = abs(rads[i] - new_rad)
    return mn


def get_dist(num, rad):
    if num == 0:
        return 2 * np.pi * rad
    res = np.sin(np.pi / num) * 2 * rad
    return res


def simpler(lens1: Metalens, lens2: Metalens):
    return len(lens1.rads) < len(lens2.rads) or sum(lens1.nums) < sum(lens2.nums)


def mutate(subject: Metalens):
    new_subject = copy(subject)
    mutation = np.random.choice(mutations)
    i = np.random.randint(0, len(subject.rads))
    # print("mutate ring", i, end=" ")
    if mutation == 'radius':
        # print("mutate radius")
        new_rad = np.random.randint(MIN_RAD, MAX_RAD)
        while get_dist_to_closest_ring(new_rad, new_subject.rads, i) < MIN_DIST or get_dist(new_subject.nums[i],
                                                                                            new_rad) < MIN_DIST:
            new_rad = np.random.randint(MIN_RAD, MAX_RAD)
        new_subject.rads[i] = new_rad
    elif mutation == 'angle':
        # print("mutate initial angle")
        new_subject.starts[i] = np.random.uniform(0, 2 * np.pi)
    elif mutation == 'number':
        # print("mutate number of particles")
        new_num = np.random.randint(0, MAX_NUM)
        while get_dist(new_num, new_subject.rads[i]) < MIN_DIST:
            new_num = np.random.randint(0, MAX_NUM)
        new_subject.nums[i] = new_num

    return new_subject


def export_as_json(subject: Metalens):
    file = open('lens/lens_focus_{}_random_seed_{}.json'.format(subject.focus, subject.random_seed), 'w+')
    file.write(jsonpickle.encode(subject))
    file.close()


def import_from_json(filepath):
    file = open(filepath, 'r')
    return jsonpickle.decode(file.read())
