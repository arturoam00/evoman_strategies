import os

import hydra
import matplotlib.pyplot as plt
import numpy as np


@hydra.main(config_name="config", config_path="../conf", version_base=None)
def main(cfg):
    # First get the root dir of the project, just in case
    current_script_path = os.path.realpath(__file__)
    root_directory = os.path.dirname(os.path.dirname(current_script_path))

    data_folder = os.path.join(root_directory, cfg.agent.data_folder)
    image_folder = os.path.join(root_directory, cfg.agent.image_folder)
    filename = os.path.join(root_directory, cfg.results.lines_filename)

    enemies = str(cfg.environment.enemies)
    id_1, id_2 = cfg.compare.evo1, cfg.compare.evo2

    path_1 = os.path.join(data_folder, f"{id_1}_enemies{enemies}.npy")
    path_2 = os.path.join(data_folder, f"{id_2}_enemies{enemies}.npy")

    # load data from data folder
    arr_dict_1 = np.load(path_1, allow_pickle=True).item()
    arr_dict_2 = np.load(path_2, allow_pickle=True).item()

    # read data
    mean_fits_1 = arr_dict_1["mean_fits"]
    max_fits_1 = arr_dict_1["max_fits"]
    mean_fits_2 = arr_dict_2["mean_fits"]
    max_fits_2 = arr_dict_2["max_fits"]

    if not os.path.exists(image_folder) and not os.path.isdir(image_folder):
        os.makedirs(image_folder)

    lines = (mean_fits_1, max_fits_1, mean_fits_2, max_fits_2)

    std = []
    for y in lines:
        std.append(np.std(y))

    # x vector spanning all generations
    x = np.arange(0, len(lines[0]))

    fig, ax = plt.subplots()

    # plot lines
    ax.plot(x, lines[0], label=f"mean fitness {id_1}")
    ax.plot(x, lines[1], label=f"max fitness {id_1}")
    ax.plot(x, lines[2], label=f"mean fitness {id_2}")
    ax.plot(x, lines[3], label=f"max fitness {id_2}")

    # plot standard deviation areas
    for y, std in zip(lines, std):
        ax.fill_between(x, y - std, y + std, alpha=0.2)

    ax.set_xlabel("Generations", fontsize=15)
    ax.set_ylabel("Fitness", fontsize=15)
    ax.set_title(f"Enemies {','.join(enemies)}", fontsize=15)
    ax.set_xlim(0, len(lines[0]))
    ax.set_ylim(min(np.min(lines), 0), 100)
    ax.legend(loc="lower right", fontsize=14)

    plt.savefig(filename, format="png")

    plt.close()


if __name__ == "__main__":
    main()
