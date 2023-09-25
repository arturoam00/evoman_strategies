import numpy as np

from demo_controller import player_controller
from environment_specialist import EnvironmentSpecialist
from evolution_specialist import EvolutionSpecialistBase


def main():
    n_hidden = 10  # neural network hidden layers

    # initializes environment
    env = EnvironmentSpecialist(
        experiment_name="specialist",
        enemies=[2],
        player_controller=player_controller(n_hidden),
    )

    # initializes evolution object
    evo = EvolutionSpecialistBase(env=env, pop_size=100, lower=-1, upper=1)

    for _ in evo.run_simulation(n_gens=100):
        print(np.mean(evo.fit_pop))


if __name__ == "__main__":
    main()
