#!/usr/bin/env python

import numpy as np
import joblib
from get_12ECG_features import get_12ECG_features

def run_12ECG_classifier(data,header_data,classes,model):

    num_classes = len(classes)
    print("num_classes")
    print(num_classes)
    current_label = np.zeros(num_classes, dtype=int)
    print("current_label")
    print(current_label)
    print(len(current_label))
    current_score = np.zeros(num_classes)
    print("current_score")
    print(current_score)
    print(len(current_score))

    # Use your classifier here to obtain a label and score for each class. 
    features=np.asarray(get_12ECG_features(data,header_data))
    print("features")
    print(features)
    print(len(features))
    feats_reshape = features.reshape(1,-1)
    print("feats_reshape")
    print(feats_reshape)
    print(len(feats_reshape))
    label = model.predict(feats_reshape)
    print("label")
    print(label)
    label[0].tolist()
    print("label2")
    print(label[0])
    score = model.predict_proba(feats_reshape)
    print("score")
    print(score)
    print(len(score))

    print(score[0][0][0])
    print(score[0][0][1])
    
    '''
    current_label[label] = 0
    print("current_label")
    print(current_label)
    print(current_label[label])
    print(len(current_label))
    '''

    for i in range(num_classes):
        print(i)
        current_score[i] = np.array(score[i][0][1])
    

    print(label[0])
    print(type(label[0]))
    print(current_score)
    print(type(current_score))

    return label[0], current_score

def load_12ECG_model():
    # load the model from disk 
    filename='finalized_model.sav'
    loaded_model = joblib.load(filename)

    return loaded_model
