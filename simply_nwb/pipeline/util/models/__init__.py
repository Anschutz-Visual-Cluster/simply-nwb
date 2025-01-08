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
    def read_modeldata(modeldata):
        byts = base64.b64decode(modeldata)
        return pickle.loads(byts)

    @staticmethod
    def get_model(model_name):
        print(f"Getting default model '{model_name}'..")
        # Only a select models are saved this way
        if model_name == "direction_model":
            from simply_nwb.pipeline.util.models.direction_model import MODEL_DATA
        elif model_name == "epoch_nasal_regressor":
            from simply_nwb.pipeline.util.models.epoch_nasal_regressor import MODEL_DATA
        elif model_name == "epoch_temporal_regressor":
            from simply_nwb.pipeline.util.models.epoch_temporal_regressor import MODEL_DATA
        elif model_name == "epoch_nasal_classifier":
            from simply_nwb.pipeline.util.models.epoch_nasal_classifier import MODEL_DATA
        elif model_name == "epoch_temporal_classifier":
            from simply_nwb.pipeline.util.models.epoch_temporal_classifier import MODEL_DATA
        else:
            raise ValueError(f"No model named '{model_name}' found!")

        return ModelReader.read_modeldata(MODEL_DATA)
