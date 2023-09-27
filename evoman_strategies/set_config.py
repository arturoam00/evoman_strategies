import json
import sys


def main(
    id_1="EA1", id_2="EA2", enemies="2", pop_size=100, n_gens=30, upper=1, lower=-1
):
    if enemies not in "12345678":
        raise ValueError(f"Invalid enemies {enemies}")

    cfg = {
        "id_1": str(id_1),
        "id_2": str(id_2),
        "enemies": str(enemies),
        "pop_size": int(pop_size),
        "n_gens": int(n_gens),
        "upper": float(upper),
        "lower": float(lower),
    }

    with open("config.json", "w") as f:
        json.dump(cfg, f)


if __name__ == "__main__":
    main(*sys.argv[1:])
