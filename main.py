import numpy as np

from evoulution_specialist import EvolutionSpecialistBase


class EvolutionSpecialist(EvolutionSpecialistBase):
    def __init__(
        self,
        experiment_name,
        pop_size=100,
        lower=-1,
        upper=1,
        n_hidden=10,
        headless=True,
    ) -> None:
        super().__init__(experiment_name, pop_size, lower, upper, n_hidden, headless)

    def mutate(self, x, prob=0.2):
        for i in range(len(x)):
            if prob >= np.random.uniform(0, 1):
                x[i] += np.random.normal(0, 1)
                x[i] = self._check_limits(x[i])
        return x


def main():
    evo = EvolutionSpecialistBase(
        experiment_name="specialist", pop_size=100, lower=-1, upper=1, n_hidden=10
    )

    for sim in evo.run_simulation(n_gens=100):
        print(max(evo.fit_pop))


if __name__ == "__main__":
    main()
