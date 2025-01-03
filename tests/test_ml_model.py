from simply_nwb.pipeline.enrichments.saccades.predict_ml_model import PredictSaccadeMLEnrichment


def main():
    prefix = "../data/extraction/nr1ko"
    PredictSaccadeMLEnrichment.retrain([
        (f"{prefix}/saccade_times.csv", f"{prefix}/20241023_unitR2_session004_rightCam_timestamps.txt", f"{prefix}/20241023_unitR2_session004_rightCam-0000DLC_resnet50_pupilsizeFeb6shuffle1_1030000.csv")
    ])

    tw = 2


if __name__ == "__main__":
    main()

