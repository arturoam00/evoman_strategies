import numpy as np

from evolution_specialist import EvolutionSpecialistBase


# this is an example of how can one modify the default behaviour
class EvolutionSpecialist(EvolutionSpecialistBase):
    def __init__(
        self,
        experiment_name,
        pop_size=100,
        lower=-1,
        upper=1,
        n_hidden=10,
        pcont=None,
        headless=True,
    ) -> None:
        super().__init__(
            experiment_name, pop_size, lower, upper, n_hidden, pcont, headless
        )

    def mutate(self, x, prob=0.0):
        for i in range(len(x)):
            if prob >= np.random.uniform(0, 1):
                x[i] += np.random.normal(0, 1)
                x[i] = self._check_limits(x[i])
        return x


def main():
    # this is the default behaviour
    evo = EvolutionSpecialist(
        experiment_name="specialist", pop_size=100, lower=-1, upper=1, n_hidden=10
    )

    for _ in evo.run_simulation(n_gens=100):
        print(np.mean(evo.fit_pop))


if __name__ == "__main__":
    main()
