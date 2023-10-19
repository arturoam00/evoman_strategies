import numpy as np

from .base_evolution import BaseEvolution


class Adaptive(BaseEvolution):
    def __init__(self, env, params) -> None:
        super().__init__(env, params)
        self.n_sigmas = 1
        self.n_vars = self._n_vars + self.n_sigmas

    @property
    def offspring_prop(self):
        return (
            self.params.lambda_0
            + self.params.a * (self.gen / self._total_gen) ** self.params.b
        )

    def initialization(self):
        population = np.random.uniform(
            self.params.lower,
            self.params.upper,
            (self.params.pop_size, self._n_vars),
        )

        step_sizes = np.random.uniform(0, 1, (self.params.pop_size, self.n_sigmas))

        return np.concatenate([population, step_sizes], axis=1)

    def _check_sigma(self, x):
        if x < self.params.epsilon_0:
            return self.params.epsilon_0
        return x

    def mutate(self, x):
        # mutate sigmas FIRST
        for i in range(self.n_sigmas):
            x[-1 - i] *= np.exp(self.params.learning_rate * np.random.normal())
            x[-1 - i] = self._check_sigma(x[-1 - i])

        # then mutate individual
        for i in range(self._n_vars):
            if self.mutation_prob > np.random.uniform():
                sigma = x[self._n_vars : self._n_vars + i + 1]
                x[i] += sigma * np.random.normal(0, 1)
                x[i] = self._check_limits(x[i])
        return x
