from copy import copy

import numpy as np
from base_evolution import BaseEvolution


class EA2(BaseEvolution):
    def select_parents(self, prop=0.5):
        return super().select_parents(prop)

    def mutate(self, x, prob=0.4):
        for i in range(len(x)):
            if prob > np.random.uniform():
                x[i] += np.random.normal(0, 1)
                x[i] = self._check_limits(x[i])
        return x
