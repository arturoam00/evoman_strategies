import numpy as np
from base_evolution import BaseEvolution


class EA1(BaseEvolution):
    def __init__(self, env, pop_size=100, lower=-1, upper=1) -> None:
        super().__init__(env, pop_size, lower, upper)

    def mutate(self, x, prob=0.4):
        for i in range(len(x)):
            if prob > np.random.uniform():
                x[i] += np.random.normal(0, 1)
                x[i] = self._check_limits(x[i])
        return x

    def select_parents(self, prop=0.5):
        return super().select_parents(prop)
