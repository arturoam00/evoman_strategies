import hydra
import numpy as np
from demo_controller import player_controller
from environment_ import Environment_
from hydra.utils import instantiate
from simulation_utils.simulator import Simulator


def save_data_specialist(env, cfg, *args):
    data_managers = args
    for d in data_managers:
        individual_gain = np.zeros(len(d.best_guys))

        # best guy analysis (testing individual gain)
        for i in range(len(d.best_guys)):
            individual_gain[i] = env.return_gain(d.best_guys[i])

        # save final results
        d.store_individual_gain(individual_gain)
        d.save_results(folder=cfg.agent.data_folder, enemy=cfg.environment.enemies)

    return


def save_data_generalist(env, cfg, *args):
    pass


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    # initialize environment
    env = Environment_(
        experiment_name=cfg.environment.name,
        enemies=cfg.environment.enemies,
        player_controller=player_controller(cfg.nn.n_hidden),
    )

    evolutions = []
    for evolution in cfg.simulation.evolutions:
        evo_config = hydra.compose(f"evolution/{evolution}")

        evo = instantiate(evo_config.evolution.create, env=env)
        evolutions.append(evo)

    # Initialize simulator and run n_sim simulations to collect average
    # fitness and max fitness data for each evolutionary strategy
    simulator = Simulator(*evolutions)
    data_managers = simulator.run(cfg.simulation.n_sim, cfg.simulation.n_gens)

    # dedicated function to save data to files
    if cfg.agent.type == "specialist":
        save_data_specialist(env, cfg, *data_managers)
    else:
        save_data_generalist(env, cfg, *data_managers)
    return


if __name__ == "__main__":
    main()
