import numpy as np
from copy import copy

mutations = ['radius', 'angle', 'number', 'add', 'remove']


class Metalens:
    def __init__(self, rads, starts, nums):
        self.rads = rads
        self.starts = starts
        self.nums = nums

    def __copy__(self):
        return Metalens(copy(self.rads), copy(self.starts), copy(self.nums))


def get_dist_to_closest_ring(new_rad, rads, index):
    mn = np.inf
    for i in range(len(rads)):
        if abs(rads[i] - new_rad) < mn and i != index:
            mn = abs(rads[i] - new_rad)
    return mn


def get_dist(num, rad):
    res = np.sin(np.pi / num) * 2 * rad
    return res


def simpler(lens1: Metalens, lens2: Metalens):
    return len(lens1.rads) < len(lens2.rads) or sum(lens2.nums) < sum(lens2.nums)


def mutate(subject: Metalens):
    new_subject = copy(subject)
    mutation = np.random.choice(mutations, 1, p=[0.3, 0.3, 0.3, 0.05, 0.05])
    if mutation == 'add':
        print("add new ring")
        new_rad = np.random.randint(1000, 15000)
        while get_dist_to_closest_ring(new_rad, new_subject.rads, -1) < 300:
            new_rad = np.random.randint(1000, 15000)
        new_angle = np.random.uniform(0, 2 * np.pi)
        new_num = np.random.randint(0, 500)
        while get_dist(new_num, new_rad) < 300:
            new_num = np.random.randint(0, 500)
        np.append(new_subject.rads, new_rad)
        np.append(new_subject.starts, new_angle)
        np.append(new_subject.nums, new_num)
        return new_subject
    i = np.random.randint(0, len(subject.rads))
    print("mutate ring", i, end=" ")
    if mutation == 'radius':
        print("mutate radius")
        new_rad = np.random.randint(1000, 15000)
        while get_dist_to_closest_ring(new_rad, new_subject.rads, i) < 300 or get_dist(new_subject.nums[i], new_rad) < 300:
            new_rad = np.random.randint(1000, 15000)
        new_subject.rads[i] = new_rad
    elif mutation == 'angle':
        print("mutate initial angle")
        new_subject.starts[i] = np.random.uniform(0, 2 * np.pi)
    elif mutation == 'number':
        print("mutate number of particles")
        new_num = np.random.randint(1, 500)
        while get_dist(new_num, new_subject.rads[i]) < 300:
            new_num = np.random.randint(1, 500)
        new_subject.nums[i] = new_num
    elif mutation == 'remove':
        print("remove ring")
        np.delete(new_subject.rads, i)
        np.delete(new_subject.starts, i)
        np.delete(new_subject.nums, i)

    return new_subject
