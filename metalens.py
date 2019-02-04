import numpy as np
from copy import copy


class Metalens:
    def __init__(self, rads, starts, nums):
        self.rads = rads
        self.starts = starts
        self.nums = nums

    def __copy__(self):
        return Metalens(copy(self.rads), copy(self.starts), copy(self.nums))


def mutate(subject: Metalens):
    new_subject = copy(subject)
    i = np.random.randint(0, len(subject.rads))
    feature = np.random.randint(0, 3)
    print("mutate ring", i, end=" ")
    if feature == 0:
        print("mutate radius")
        new_subject.rads[i] += np.random.randint(-new_subject.rads[i], 2001)
    elif feature == 1:
        print("mutate initial angle")
        new_subject.starts[i] += np.random.uniform(0, 2 * np.pi)
    else:
        print("mutate number of particles")
        new_subject.nums[i] += np.random.randint(-new_subject.nums[i], 100)
    return new_subject
