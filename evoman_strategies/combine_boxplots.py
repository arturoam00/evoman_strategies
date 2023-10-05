import os
import sys
import json

import matplotlib.pyplot as plt
import numpy as np


def combine_boxplots(
    *arg, filename="combined_boxplots.png", data_folder="data", image_folder="images"
):
    with open("config.json", "r") as f:
        cfg = json.load(f)
        
    id_1, id_2, *_ = cfg.values()
    
    enemies = "".join(map(str, arg))
    results = []
    labels = []

    for enemy in enemies:
        path_1 = os.path.join(data_folder, f"{id_1}_enemy{enemy}.npy")
        path_2 = os.path.join(data_folder, f"{id_2}_enemy{enemy}.npy")

        arr_dict_1 = np.load(path_1, allow_pickle=True).item()
        arr_dict_2 = np.load(path_2, allow_pickle=True).item()

        indi_gain_1 = arr_dict_1["individual_gain"]
        indi_gain_2 = arr_dict_2["individual_gain"]

        if not os.path.exists(image_folder) and not os.path.isdir(image_folder):
            os.mkdir(image_folder)

        results.extend([indi_gain_1, indi_gain_2])
        labels.extend([f"{id_1} E{enemy}", f"{id_2} E{enemy}"])

    fig, ax = plt.subplots()

    ax.boxplot(results, labels=labels)
    ax.set_ylabel("Individual gain", fontsize=15)

    plt.savefig(os.path.join(image_folder, filename), format="png")


if __name__ == "__main__":
    enemies = "".join(map(str, sys.argv[1:]))
    combine_boxplots(*sys.argv[1:], filename=f"combined_boxplots_{enemies}.png")
