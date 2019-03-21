import numpy as np
from copy import copy

# mutations = ['radius', 'angle', 'number', 'add', 'remove']
mutations = ['radius', 'angle', 'number']
true_or_false = [True, False]

MAX_NUM = 100
MIN_RAD = 1000
MAX_RAD = 15000
MIN_DIST = 300


class Metalens:
    def __init__(self, rads, starts, nums):
        self.rads = rads
        self.starts = starts
        self.nums = nums
        self.score = np.inf

    def __copy__(self):
        return Metalens(copy(self.rads), copy(self.starts), copy(self.nums))

    def __lt__(self, other):
        return self.score < other.score


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


def breed(first: Metalens, second: Metalens):
    lenses = [first, second]
    rads = []
    starts = []
    nums = []
    for i in range(len(first.rads)):
        ind = np.random.randint(0, 2)
        rads.append(lenses[ind].rads[i])
        starts.append(lenses[ind].starts[i])
        nums.append(lenses[ind].nums[i])
    res = Metalens(rads, starts, nums)
    return res


def is_alive(lens: Metalens):
    n = len(lens.rads)
    for i in range(n):
        if get_dist_to_closest_ring(lens.rads[i], lens.rads, i) < MIN_DIST:
            return False
    for i in range(n):
        if get_dist(lens.nums[i], lens.rads[i]) < MIN_DIST:
            return False
    if sum(lens.nums) > MAX_NUM * n:
        return False
    return True


def breed_n(population: [Metalens], n):
    res = []
    while len(res) <= n:
        lenses = np.random.choice(population, 2)
        new_subject = breed(lenses[0], lenses[1])
        if is_alive(new_subject):
            res.append(new_subject)
    return res


def mutate_n(population: [Metalens], n):
    lenses_to_mutate = np.random.choice(population, n)
    res = []
    for lens in lenses_to_mutate:
        new_lens = mutate(lens)
        while not is_alive(new_lens):
            new_lens = mutate(lens)
        res.append(new_lens)
    return res


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
    return len(lens1.rads) < len(lens2.rads) or sum(lens2.nums) < sum(lens2.nums)


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
