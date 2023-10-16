import os
from collections import namedtuple

import hydra
import matplotlib.pyplot as plt
import numpy as np

Line = namedtuple("Line", ["id", "label", "values", "std"])


@hydra.main(config_name="config", config_path="../conf", version_base=None)
def main(cfg):
    # First get the root dir of the project, just in case
    current_script_path = os.path.realpath(__file__)
    root_directory = os.path.dirname(os.path.dirname(current_script_path))

    # what are we plotting
    evolutions = cfg.simulation.evolutions
    enemies = str(cfg.environment.enemies)

    # where from and where to are we plotting it
    data_folder = os.path.join(root_directory, cfg.agent.data_folder)
    image_folder = os.path.join(root_directory, cfg.agent.image_folder)
    filename = os.path.join(
        root_directory,
        cfg.agent.image_folder,
        f"lplot_{'_'.join(evolutions)}_e{enemies}",
    )

    if not os.path.exists(image_folder) and not os.path.isdir(image_folder):
        os.makedirs(image_folder)

    # plot it
    fig, ax = plt.subplots()

    lines = []

    for evo in evolutions:
        path = os.path.join(data_folder, f"{evo}_enemies{enemies}.npy")
        arr_dict = np.load(path, allow_pickle=True).item()
        mean_fits = arr_dict["mean_fits"]
        max_fits = arr_dict["max_fits"]
        lines.extend(
            [
                Line(
                    id=evo, label="mean fits", values=mean_fits, std=np.std(mean_fits)
                ),
                Line(id=evo, label="max fits", values=max_fits, std=np.std(max_fits)),
            ]
        )

    # x vector spanning all generations
    gens = len(lines[0].values)
    x = np.arange(0, gens)

    for line in lines:
        ax.plot(x, line.values, label=f"{line.label} {line.id}")
        ax.fill_between(x, line.values - line.std, line.values + line.std, alpha=0.2)

    ax.set_xlabel("Generations", fontsize=15)
    ax.set_ylabel("Fitness", fontsize=15)
    ax.set_title(f"Enemies {','.join(enemies)}", fontsize=15)
    ax.set_xlim(0, gens)
    ax.set_ylim(min(np.min([line.values for line in lines]), 0), 100)
    ax.legend(loc="lower right", fontsize=14)

    plt.savefig(filename + ".png", format="png")

    plt.close()


if __name__ == "__main__":
    main()
