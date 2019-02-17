import numpy as np
from copy import copy

# mutations = ['radius', 'angle', 'number', 'add', 'remove']
mutations = ['radius', 'angle', 'number']
true_or_false = [True, False]


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
    rads = np.array([np.random.randint(1000, 15000) for _ in range(n)])
    starts = np.array([np.random.uniform(0, 2 * np.pi) for _ in range(n)])
    nums = np.array([0] * n)
    for i in range(n):
        new_num = np.random.randint(0, 500)
        while get_dist(new_num, rads[i]) < 1000:
            new_num = np.random.randint(0, 500)
        nums[i] = new_num
    return Metalens(rads, starts, nums)


def crossover(subject1: Metalens, subject2: Metalens):
    i = np.random.randint(0, len(subject1.rads))
    child1 = Metalens(rads=np.append(subject1.rads[:i], subject2.rads[i:]),
                      nums=np.append(subject1.nums[:i], subject2.nums[i:]),
                      starts=np.append(subject1.starts[:i], subject2.starts[i:]))
    child2 = Metalens(rads=np.append(subject2.rads[:i], subject1.rads[i:]),
                      nums=np.append(subject2.nums[:i], subject1.nums[i:]),
                      starts=np.append(subject2.starts[:i], subject1.starts[i:]))
    if np.random.choice(true_or_false, 1, p=[0.3, 0.7]):
        child1 = mutate(child1)
    if np.random.choice(true_or_false, 1, p=[0.3, 0.7]):
        child2 = mutate(child2)
    return child1, child2


def breed(population: [Metalens]):
    np.random.shuffle(population)
    n = len(population)
    subjects1 = population[:n // 2]
    subjects2 = population[n // 2:]
    assert len(subjects1) == len(subjects2)
    for i in range(n // 2):
        child1, child2 = crossover(subjects1[i], subjects2[i])
        population = np.concatenate((population, [child1, child2]))
    return population


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
    # mutation = np.random.choice(mutations, 1, p=[0.3, 0.3, 0.3, 0.05, 0.05])
    mutation = np.random.choice(mutations)
    # if mutation == 'add':
    #     print("add new ring")
    #     new_rad = np.random.randint(1000, 15000)
    #     while get_dist_to_closest_ring(new_rad, new_subject.rads, -1) < 300:
    #         new_rad = np.random.randint(1000, 15000)
    #     new_angle = np.random.uniform(0, 2 * np.pi)
    #     new_num = np.random.randint(0, 500)
    #     while get_dist(new_num, new_rad) < 300:
    #         new_num = np.random.randint(0, 500)
    #     np.append(new_subject.rads, new_rad)
    #     np.append(new_subject.starts, new_angle)
    #     np.append(new_subject.nums, new_num)
    #     return new_subject
    i = np.random.randint(0, len(subject.rads))
    # print("mutate ring", i, end=" ")
    if mutation == 'radius':
        # print("mutate radius")
        new_rad = np.random.randint(1000, 15000)
        while get_dist_to_closest_ring(new_rad, new_subject.rads, i) < 300 or get_dist(new_subject.nums[i],
                                                                                       new_rad) < 300:
            new_rad = np.random.randint(1000, 15000)
        new_subject.rads[i] = new_rad
    elif mutation == 'angle':
        # print("mutate initial angle")
        new_subject.starts[i] = np.random.uniform(0, 2 * np.pi)
    elif mutation == 'number':
        # print("mutate number of particles")
        new_num = np.random.randint(0, 500)
        while get_dist(new_num, new_subject.rads[i]) < 300:
            new_num = np.random.randint(0, 500)
        new_subject.nums[i] = new_num
    # elif mutation == 'remove':
    #     print("remove ring")
    #     np.delete(new_subject.rads, i)
    #     np.delete(new_subject.starts, i)
    #     np.delete(new_subject.nums, i)

    return new_subject
