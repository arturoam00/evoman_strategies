import numpy as np

from .base_evolution import BaseEvolution


class FPS(BaseEvolution):
    def mutate(self, x):
        prob = self.params.get("mutation_prob", self.mutation_prob)
        for i in range(len(x)):
            if prob > np.random.uniform():
                x[i] += np.random.normal(0, 1)
                x[i] = self._check_limits(x[i])
        return x
