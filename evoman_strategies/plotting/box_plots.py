import os
from collections import namedtuple
from collections.abc import Iterable
from random import choice, seed

import hydra
import matplotlib.pyplot as plt
import numpy as np

seed(123)

Box = namedtuple("Box", ["data", "enemies", "ea", "color"])


def rand_color():
    return "#" + "".join([choice("0123456789ABCDEF") for _ in range(6)])


def enemies_label(e):
    return f"[{','.join(str(e))}]"


@hydra.main(config_name="config", config_path="../conf", version_base=None)
def main(cfg):
    # First get the root dir of the project, just in case
    current_script_path = os.path.realpath(__file__)
    root_directory = os.path.dirname(os.path.dirname(current_script_path))

    # what are we plotting
    evolutions = cfg.simulation.evolutions
    enemies = cfg.environment.enemies

    if not isinstance(evolutions, Iterable) or isinstance(evolutions, str):
        raise ValueError("evolutions be lists")

    if not isinstance(enemies, Iterable) or isinstance(evolutions, str):
        raise ValueError("enemies be lists")

    # where from and where to are we plotting it
    data_folder = os.path.join(root_directory, cfg.agent.data_folder)
    image_folder = os.path.join(root_directory, cfg.agent.image_folder)
    filename = os.path.join(
        root_directory,
        cfg.agent.image_folder,
        f"bplot_{'_'.join(evolutions)}_e{'_'.join(str(e) for e in enemies)}",
    )

    if not os.path.exists(image_folder) and not os.path.isdir(image_folder):
        os.makedirs(image_folder)

    # plot
    fig, ax = plt.subplots()

    boxes = []

    for e in enemies:
        color = rand_color()

        for evo in evolutions:
            path = os.path.join(data_folder, f"{evo}_enemies{e}.npy")
            arr_dict = np.load(path, allow_pickle=True).item()
            indi_gains = arr_dict["individual_gain"]
            boxes.append(
                Box(data=indi_gains, enemies=enemies_label(e), ea=evo, color=color)
            )

    bx = ax.boxplot(
        x=[box.data for box in boxes],
        labels=[box.ea for box in boxes],
        patch_artist=True,
    )

    colors = [b.color for b in boxes]
    for i, c in enumerate(colors):
        bx["boxes"][i].set_facecolor(c)

    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color=color, label=enemies_label(e))
        for e, color in zip(enemies, colors[:: len(evolutions)])
    ]

    ax.set_ylabel("individual gain", fontsize=15)
    ax.legend(handles=legend_elements, fontsize=15)
    ax.set_xticklabels([box.ea for box in boxes], fontsize=12)

    plt.savefig(filename + ".png", format="png")

    plt.close()


if __name__ == "__main__":
    main()
