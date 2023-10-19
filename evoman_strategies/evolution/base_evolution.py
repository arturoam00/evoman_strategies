from copy import copy

import numpy as np


class BaseEvolution:
    def __init__(
        self,
        env,
        params,
    ) -> None:
        self.env = env
        self.params = params
        _n_hidden = self.env.player_controller.n_hidden[0]
        self._n_vars = (self.env.get_num_sensors() + 1) * _n_hidden + (
            _n_hidden + 1
        ) * 5  # this is the actual length of an individual
        self.n_vars = self._n_vars  # this can be used to introduce evolving σs

        self._pop = None
        self.fit_pop = None
        self.gen = None
        self.parents = None
        self.offspring = None
        self._total_gen = None

    @property
    def offspring_prop(self):
        return self.params.offspring_prop

    @property
    def mutation_prob(self):
        return self.params.mutation_prob

    @property
    def mutation_ssize(self):
        return self.params.mutation_ssize

    @property
    def parent_prop(self):
        return self.params.parent_prop

    @property
    def pop(self):
        return self._pop[:, : self._n_vars]

    @property
    def fittest_individual(self):
        return self.pop[np.argsort(-self.fit_pop)][0]

    @property
    def max_fitness(self):
        return np.max(self.fit_pop)

    def evaluate(self, pop):
        return self.env.evaluate(pop[:, : self._n_vars])

    def _check_limits(self, x):
        if x < self.params.lower:
            return self.params.lower
        elif x > self.params.upper:
            return self.params.upper
        return x

    def norm(self, x):
        c = self.params.get("sigma_scaling", 2)
        # this is called `sigma scaling`
        for i in range(len(x)):
            x[i] = max(x[i] - (np.mean(self.fit_pop) - c * np.std(self.fit_pop)), 1e-15)
        return x / np.sum(x)

    def initialization(self):
        return np.random.uniform(
            self.params.lower,
            self.params.upper,
            (self.params.pop_size, self.n_vars),
        )

    def select_parents(self):
        if self.fit_pop is None:
            self.fit_pop = self.evaluate(self._pop)

        # this is important (to copy self.fit_pop) so as to not change the fitness values
        fps = self.norm(copy(self.fit_pop))
        parents = np.random.choice(
            np.arange(self.params.pop_size),
            size=int(self.params.pop_size * self.parent_prop),
            p=fps,
            replace=False,
        )
        return self._pop[parents]

    def mutate(self, x):
        for i in range(self._n_vars):
            if self.mutation_prob > np.random.uniform():
                x[i] += np.random.normal(0, self.mutation_ssize)
                x[i] = self._check_limits(x[i])
        return x

    def calculate_offspring(self):
        n_parents = np.size(self.parents, 0)
        n_offspring = int(self.params.pop_size * self.offspring_prop)
        offspring = np.zeros((n_offspring, self.n_vars))

        for i in range(0, n_offspring - 1, 2):
            # select two parents out of all possible parents
            p1 = self.parents[np.random.randint(0, n_parents)]
            p2 = self.parents[np.random.randint(0, n_parents)]

            # choose the recombination parameter ɑ
            alpha = np.random.uniform(0, 1)

            # recombine
            offspring[i] = alpha * p1 + (1 - alpha) * p2
            offspring[i + 1] = alpha * p2 + (1 - alpha) * p1

            # mutate
            offspring[i] = self.mutate(offspring[i])
            offspring[i + 1] = self.mutate(offspring[i + 1])

        return offspring

    def selection(self):
        fit_pop_total = np.concatenate((self.fit_pop, self.evaluate(self.offspring)))
        total = np.vstack((self._pop, self.offspring))
        self.fit_pop = np.array(
            sorted(fit_pop_total, reverse=True)[: self.params.pop_size]
        )
        return total[np.argsort(-fit_pop_total)][: self.params.pop_size]

    def run_simulation(self, n_gens):
        self._total_gen = n_gens
        self._pop = self.initialization()

        for gen in range(n_gens):
            self.gen = gen
            self.parents = self.select_parents()
            self.offspring = self.calculate_offspring()
            self._pop = self.selection()
            yield

    def run_all(self, n_gens):
        sim = self.run_simulation(n_gens)
        while True:
            try:
                next(sim)
            except StopIteration:
                return

    def restore(self):
        self._pop = None
        self.fit_pop = None
        self.parents = None
        self.offspring = None

    def __str__(self) -> str:
        return self.__class__.__name__
