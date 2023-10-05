from copy import copy

import numpy as np
from base_evolution import BaseEvolution


def indexes_closest_to_mean(x, n_values):
    # first sort the array in *descending* order
    x = np.sort(x)[::-1]
    mean = np.mean(x)
    closest_idx = np.argmin(np.abs(x - mean))
    left_idx = max(closest_idx - n_values, 0)
    right_idx = min(closest_idx + n_values, len(x) - 1)
    return left_idx, right_idx


class EA2(BaseEvolution):
    def select_parents(self, prop=0.5):
        if self.fit_pop is None:
            self.fit_pop = self.env.evaluate(self.pop)

        # select fitness indexes next to fitness mean (1/2 of the population by default)
        l_idx, r_idx = indexes_closest_to_mean(
            copy(self.fit_pop), int(prop * self.pop_size // 2)
        )
        return self.pop[np.argsort(-self.fit_pop)][l_idx:r_idx]

    def mutate(self, x, prob=0.4):
        for i in range(len(x)):
            if prob > np.random.uniform():
                x[i] += np.random.normal(0, 1)
                x[i] = self._check_limits(x[i])
        return x
