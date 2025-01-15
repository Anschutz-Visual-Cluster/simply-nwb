import base64
import pickle


class ModelSaver(object):
    @staticmethod
    def save_model(filename, model_object):
        byts = pickle.dumps(model_object)
        b64 = base64.b64encode(byts)
        with open(filename, "w") as f:
            f.write(f"MODEL_DATA = {str(b64)}")


class ModelReader(object):

    @staticmethod
    def load_from_file(filename):
        if filename.endswith(".py"):
            with open(filename, "r") as f:
                data = "\n".join(f.readlines())
                b64 = data[len("MODEL_DATA = '"):-len("'")]
                return ModelReader.read_modeldata(b64)
        else:
            with open(filename, "rb") as f:
                return pickle.load(f)

    @staticmethod
    def read_modeldata(modeldata):
        byts = base64.b64decode(modeldata)
        return pickle.loads(byts)

    @staticmethod
    def get_model(model_name):
        print(f"Getting default model '{model_name}'..")
        # Only a select models are saved this way
        if model_name == "direction_model":
            from simply_nwb.pipeline.util.models.direction_model import MODEL_DATA
        elif model_name == "nasal_epoch_regressor":
            from simply_nwb.pipeline.util.models.nasal_epoch_regressor import MODEL_DATA
        elif model_name == "temporal_epoch_regressor":
            from simply_nwb.pipeline.util.models.temporal_epoch_regressor import MODEL_DATA
        elif model_name == "nasal_epoch_transformer":
            from simply_nwb.pipeline.util.models.nasal_epoch_transformer import MODEL_DATA
        elif model_name == "temporal_epoch_transformer":
            from simply_nwb.pipeline.util.models.temporal_epoch_transformer import MODEL_DATA
        else:
            raise ValueError(f"No model named '{model_name}' found!")

        return ModelReader.read_modeldata(MODEL_DATA)
