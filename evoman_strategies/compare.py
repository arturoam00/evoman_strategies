import hydra
import numpy as np
from compare_utils.comparer import Comparer
from demo_controller import player_controller
from environment_ import Environment_
from hydra.utils import instantiate


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    # initialize environment
    env = Environment_(
        experiment_name=cfg.environment.name,
        enemies=cfg.environment.enemies,
        player_controller=player_controller(cfg.nn.n_hidden),
    )

    evo1_config = hydra.compose(f"evolution/{cfg.compare.evo1}")
    evo2_config = hydra.compose(f"evolution/{cfg.compare.evo2}")

    # initialize evolution objects
    evo1 = instantiate(evo1_config.evolution.create, env=env)
    evo2 = instantiate(evo2_config.evolution.create, env=env)

    # initialize comparer and compare
    # dm1 and dm2 are the data objects for each algorithm (their results)
    comp = Comparer(evo1, evo2)
    dm1, dm2 = comp.compare(cfg.compare.n_sim, cfg.compare.n_gens)

    # TODO: this should be adjusted to just happen when environment.type == specialist
    for d in [dm1, dm2]:
        individual_gain = np.zeros(len(d.best_guys))

        # best guy analysis (testing individual gain)
        for i in range(len(d.best_guys)):
            individual_gain[i] = env.return_gain(d.best_guys[i])

        # save final results
        d.store_individual_gain(individual_gain)
        d.save_results(folder=cfg.agent.data_folder, enemy=cfg.environment.enemies)


if __name__ == "__main__":
    main()
