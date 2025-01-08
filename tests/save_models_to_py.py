import os
import pickle

from simply_nwb.pipeline.util.models import ModelSaver


def main():
    names = [
        "direction_model",
        "epoch_nasal_regressor",
        "epoch_temporal_regressor",
        "epoch_nasal_classifier",
        "epoch_temporal_classifier"
    ]

    os.chdir("..")
    for name in names:
        print(f"Processing '{name}'..")
        with open(f"{name}.pickle", "rb") as f:
            data = pickle.load(f)
            ModelSaver.save_model(f"{name}.py", data)

    print("Now put the .py files in simply_nwb/pipeline/util/models")


if __name__ == "__main__":
    main()
