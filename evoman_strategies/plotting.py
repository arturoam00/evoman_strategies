import json
import os

import matplotlib.pyplot as plt
import numpy as np


def plot_lines(y1, y2, y3, y4, filename):
    # calculate standard deviations
    std_1 = np.std(y1)
    std_2 = np.std(y2)
    std_3 = np.std(y3)
    std_4 = np.std(y4)

    # x vector spanning all generations
    x = np.arange(0, len(y1))

    fig, ax = plt.subplots()

    # plot lines
    ax.plot(x, y1, label="mean EA1")
    ax.plot(x, y2, label="max EA1")
    ax.plot(x, y3, label="mean EA2")
    ax.plot(x, y4, label="max EA2")

    # plot standard deviation areas
    ax.fill_between(x, y1 - std_1, y1 + std_1, alpha=0.2)
    ax.fill_between(x, y2 - std_2, y2 + std_2, alpha=0.2)
    ax.fill_between(x, y3 - std_3, y3 + std_3, alpha=0.2)
    ax.fill_between(x, y4 - std_4, y4 + std_4, alpha=0.2)

    ax.set_xlabel("generations")
    ax.set_ylabel("fitness")
    ax.legend()

    plt.savefig(filename, format="png")

    plt.close()


def plot_boxes(a, b, filename):
    fig, ax = plt.subplots()

    ax.boxplot([a, b], labels=["EA1", "EA2"])

    ax.set_xlabel("")
    ax.set_ylabel("individual_gain")
    ax.set_title("Individual gain")

    plt.savefig(filename, format="png")

    plt.close()


def main(data_folder="data"):
    image_folder = "images"

    with open("config.json", "r") as f:
        cfg = json.load(f)

    id_1, id_2, enemies, *_ = cfg.values()

    path_1 = os.path.join(data_folder, f"{id_1}_enemy{enemies}.npy")
    path_2 = os.path.join(data_folder, f"{id_2}_enemy{enemies}.npy")

    # load data from data folder
    arr_dict_1 = np.load(path_1, allow_pickle=True).item()
    arr_dict_2 = np.load(path_2, allow_pickle=True).item()

    # read data
    mean_fits_1 = arr_dict_1["mean_fits"]
    max_fits_1 = arr_dict_1["max_fits"]
    indi_gain_1 = arr_dict_1["individual_gain"]
    mean_fits_2 = arr_dict_2["mean_fits"]
    max_fits_2 = arr_dict_2["max_fits"]
    indi_gain_2 = arr_dict_2["individual_gain"]

    if not os.path.exists(image_folder) and not os.path.isdir(image_folder):
        os.mkdir(image_folder)

    plot_lines(
        mean_fits_1,
        max_fits_1,
        mean_fits_2,
        max_fits_2,
        os.path.join(image_folder, f"line_plot_e{enemies}.png"),
    )

    plot_boxes(
        indi_gain_1, indi_gain_2, os.path.join(image_folder, f"box_plot_e{enemies}.png")
    )


if __name__ == "__main__":
    main()
