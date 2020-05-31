#!/usr/bin/env python

import numpy as np
import joblib
from get_12ECG_features import get_12ECG_features

def run_12ECG_classifier(data, header_data, classes, model):
    
    num_classes = len(classes)

    current_label = np.zeros(num_classes, dtype=int)

    current_score = np.zeros(num_classes)

    features = np.asarray(get_12ECG_features(data, header_data))

    # feats_reshape = features
    feats_reshape = features.reshape(1, -1)

    feats_reshape = np.nan_to_num(feats_reshape)
    # feats_reshape = feats_reshape.tolist()
    scaler = joblib.load('std_scaler.bin')
    cleanFeatures = scaler.transform(feats_reshape)

    label = model.predict(cleanFeatures)

    label[0].tolist()

    score = model.predict_proba(cleanFeatures)

    for i in range(num_classes):
        current_score[i] = np.array(score[i][0][1])

    return label[0], current_score


def load_12ECG_model():
    # load the model from disk 
    filename = 'finalized_model.sav'
    loaded_model = joblib.load(filename)

    return loaded_model
