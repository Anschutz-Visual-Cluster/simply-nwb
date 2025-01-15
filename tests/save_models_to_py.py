import os
import pickle

from simply_nwb.pipeline.util.models import ModelSaver


def main():
    # prefix = ""
    prefix = "predict_gui_"

    names = [
        "nasal_epoch_regressor",
        "temporal_epoch_regressor",
        "nasal_epoch_transformer",
        "temporal_epoch_transformer"
    ]

    os.chdir("..")
    for name in names:
        name = prefix + name
        print(f"Processing '{name}'..")
        try:
            with open(f"{name}.pickle", "rb") as f:
                data = pickle.load(f)
                ModelSaver.save_model(f"{name[len(prefix):]}.py", data)
        except FileNotFoundError as e:
            print(f"File '{name}' not found! Files in cwd: '{os.listdir()}'")
            raise e

    print("Now put the generated .py files in simply_nwb/pipeline/util/models!!!!!!!!!!!!!!!!!")


if __name__ == "__main__":
    main()
