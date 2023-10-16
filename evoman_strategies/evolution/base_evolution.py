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
        ) * 5
        self.pop = np.random.uniform(
            self.params.lower,
            self.params.upper,
            (self.params.pop_size, self._n_vars),
        )
        # simulation related attrs
        self.fit_pop = None
        self.gen = None
        self.parents = None
        self.offspring = None
        self._total_gen = None

    @property
    def offspring_prop(self):
        pass

    @property
    def mutation_prob(self):
        pass

    @property
    def parent_prop(self):
        pass

    def _check_limits(self, x):
        if x < self.params.lower:
            return self.params.lower
        elif x > self.params.upper:
            return self.params.upper
        return x

    def norm(self, x):
        c = self.params.get("sscaling_param", 2)
        # this is called `sigma scaling`
        for i in range(len(x)):
            x[i] = max(x[i] - (np.mean(self.fit_pop) - c * np.std(self.fit_pop)), 1e-15)
        return x / sum(x)

    def select_parents(self):
        prop = self.params.get("parent_prop", self.parent_prop)
        if self.fit_pop is None:
            self.fit_pop = self.env.evaluate(self.pop)

        # this is important (to copy self.fit_pop) so as to not change the fitness values
        fps = self.norm(copy(self.fit_pop))
        parents = np.random.choice(
            np.arange(self.params.pop_size),
            size=int(self.params.pop_size * prop),
            p=fps,
            replace=False,
        )
        return self.pop[parents]

    def mutate(self, x):
        prob = self.params.get("mutation_prob", self.mutation_prop)
        for i in range(len(x)):
            if prob > np.random.uniform():
                x[i] += np.random.normal(0, 1)
                x[i] = self._check_limits(x[i])
        return x

    def calculate_offspring(self):
        n_parents = np.size(self.parents, 0)
        n_offspring = int(
            self.params.pop_size
            * self.params.get("offspring_prop", self.offspring_prop)
        )
        offspring = np.zeros((n_offspring, self._n_vars))

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
        fit_pop_total = np.concatenate(
            (self.fit_pop, self.env.evaluate(self.offspring))
        )
        total = np.vstack((self.pop, self.offspring))
        self.fit_pop = np.array(
            sorted(fit_pop_total, reverse=True)[: self.params.pop_size]
        )
        return total[np.argsort(-fit_pop_total)][: self.params.pop_size]

    def run_simulation(self, n_gens):
        self._total_gen = n_gens
        for gen in range(n_gens):
            self.gen = gen
            self.parents = self.select_parents()
            self.offspring = self.calculate_offspring()
            self.pop = self.selection()
            yield

    def run_all(self, n_gens):
        sim = self.run_simulation(n_gens)
        while True:
            try:
                next(sim)
            except StopIteration:
                return

    def restore(self):
        self.pop = np.random.uniform(
            self.params.lower,
            self.params.upper,
            (self.params.pop_size, self._n_vars),
        )
        self.fit_pop = None
        self.parents = None
        self.offspring = None

    def __str__(self) -> str:
        return self.__class__.__name__
