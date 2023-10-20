import os
from collections import namedtuple
from collections.abc import Iterable
from random import choice, seed

import hydra
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind

seed(123)

Box = namedtuple("Box", ["data", "enemies", "ea", "color"])


def rand_color():
    return "#" + "".join([choice("0123456789ABCDEF") for _ in range(6)])


def enemies_label(e):
    return f"[{','.join(str(e))}]"


def test_means(a, b):
    t, p = ttest_ind(a, b)
    return f"T-value: {np.round(t, 3)}, p-value: {np.round(p, 3)}"


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
        _indi_gains = []

        for evo in evolutions:
            path = os.path.join(data_folder, f"{evo}_enemies{e}.npy")
            arr_dict = np.load(path, allow_pickle=True).item()
            indi_gains = arr_dict["individual_gain"] / 8
            boxes.append(
                Box(data=indi_gains, enemies=enemies_label(e), ea=evo, color=color)
            )
            _indi_gains.append(indi_gains)
            print(
                f"Average individual gain for E.A. {evo} trained against enemies {e}: "
                + f"{np.round(np.mean(indi_gains), 2)} ± {np.round(np.std(indi_gains), 2)}"
            )

        # statistical test of the individual gains bet
        if len(_indi_gains) == 2:
            print(
                f"Restults for the t-test comparing {evolutions} for enemies {e}: "
                + f"{test_means(*_indi_gains)}"
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
