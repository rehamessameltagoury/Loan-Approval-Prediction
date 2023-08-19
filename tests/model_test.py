import os

import numpy as np
from sklearn.metrics import accuracy_score

import Loan_Approval_Predictions as model

data_path = "./output"


def read_testdata():
    X_val, y_val = model.load_pickle(os.path.join(data_path, "val_orch.pkl"))
    return X_val, y_val


def test_predictAccuracy():
    # testinng accuracy if it will be accpted or not
    rf_model = model.load_pickle("models/rf.b")
    features_xval, features_yval = read_testdata()
    actual_prediction = rf_model.predict(features_xval)
    predictedacc = accuracy_score(features_yval, actual_prediction) * 100
    assert predictedacc >= 90


def test_predict():
    # testing the first row prediction is correct
    rf_model = model.load_pickle("models/rf.b")
    features_xval, features_yval = read_testdata()
    actual_prediction = rf_model.predict(np.array([features_xval.iloc[0]]))
    assert actual_prediction == features_yval.iloc[0]
