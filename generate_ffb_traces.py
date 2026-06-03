
import torch
import argparse
import math
import numpy as np
import pandas as pd
from ffb.utils import PandasDataSet, seed_everything
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys, os, pathlib, time
sys.path.append('ffb')

from ffb.dataset import load_adult_data, load_german_data, load_compas_data, load_bank_marketing_data
from ffb.utils import InfiniteDataLoader

import networks

torch.serialization.add_safe_globals([networks.MLP])


REFERENCE_DP = {
    "adult, race" : -0.136069,
    "adult, sex" : -0.1989,
    "compas, race" : 0.097456,
    "compas, sex" : -0.12799,
    "bank_marketing, age" : 0.130428,
    "german, age" : -0.149447,
    "german, sex" : -0.074801,
}

def preprocess(ml_model, dataset, sensitive_attr, debug=0):

    # net = torch.load(ml_model)
    net = torch.load(ml_model, weights_only=False)

    if dataset == "adult":
        if debug > 0:
            print(f"Dataset: adult")
        X, y, s = load_adult_data(path="ffb_datasets/adult/raw", sensitive_attribute=sensitive_attr)

    elif dataset == "german":
        if debug > 0:
            print(f"Dataset: german")
        X, y, s = load_german_data(path="ffb_datasets/german/raw", sensitive_attribute=sensitive_attr)

    elif dataset == "compas":
        if debug > 0:
            print(f"Dataset: compas")
        X, y, s = load_compas_data(path="ffb_datasets/compas/raw", sensitive_attribute=sensitive_attr)

    elif dataset == "bank_marketing":
        if debug > 0:
            print(f"Dataset: bank_marketing")
        X, y, s = load_bank_marketing_data(path="ffb_datasets/bank_marketing/raw", sensitive_attribute=sensitive_attr)

    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    

    ml_algo = ml_model.split('/')[-1].split('_')[2 + len(dataset.split('_'))].split('.')[0]
    categorical_cols = X.select_dtypes("string").columns
    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols)

    n_features = X.shape[1]
    n_classes = len(np.unique(y))

    X_train, X, y_train, y, s_train, s = train_test_split(X, y, s, test_size=0.6, stratify=y, random_state=None) # Todo: seed random state
    
    numurical_cols = X.select_dtypes("float32").columns
    if len(numurical_cols) > 0:

        def scale_df(df, scaler):
            return pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)

        scaler = StandardScaler().fit(X[numurical_cols])

        def scale_df(df, scaler):
            return pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)

        X[numurical_cols] = X[numurical_cols].pipe(scale_df, scaler)
    
    data_loader = PandasDataSet(X, y, s)

    return net, ml_algo, data_loader


def main():
    ml_model = "experimental-results/MLmodels/model_adult_race_hsic.pkl"
    dataset = "adult"
    sensitive_attr = "race"
    ml_algo = "hsic"
    ml_model = f"experimental-results/MLmodels/model_{dataset}_{sensitive_attr}_{ml_algo}.pkl"

    net, ml_algo, data_loader = preprocess(ml_model, dataset, sensitive_attr)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log_gAseen = []
    log_gAacc = []
    log_gBseen = []
    log_gBacc = []
    log_score = []
    log_prediction = []
    log_ground_truth = []
    log_group = []
    log_step = []
    log_event = []

    gAseen = 0
    gAacc = 0
    gBseen = 0
    gBacc = 0

    max_step = 10000

    for step, (x, y, s) in enumerate(data_loader):
        if step > max_step:
            break
        if ml_algo == "laftr":
            h, decoded, output, adv_pred = net(x.to(device), s.to(device))
        else:
            h, output = net(x.to(device))
        score = output.detach().cpu().numpy()[0]
        net_proposes_accept = score > 0.5
        if (s == 0): # Group A
            gAseen += 1
            if net_proposes_accept:
                gAacc += 1
                the_event = "A_acc"
            else:
                the_event = "A_rej"
        else:
            gBseen += 1
            if net_proposes_accept:
                gBacc += 1
                the_event = "B_acc"
            else:
                the_event = "B_rej"

        log_gAseen.append(gAseen)
        log_gAacc.append(gAacc)
        log_gBseen.append(gBseen)
        log_gBacc.append(gBacc)
        log_score.append(score)
        if net_proposes_accept:
            log_prediction.append(1)
        else:
            log_prediction.append(0)

        log_ground_truth.append(y.detach().cpu().numpy()[0])
        log_group.append(s.detach().cpu().numpy()[0])
        log_step.append(step)
        log_event.append(the_event)


    res_dict = {
        "time" : log_step,
        "input" : log_event,
        "gAseen" : log_gAseen,
        "gAacc" : log_gAacc,
        "gBseen" : log_gBseen,
        "gBacc" : log_gBacc,
        "score" : log_score,
        "prediction" : log_prediction,
        "ground_truth" : log_ground_truth,
        "group_membership" : log_group
    }

    res_df = pd.DataFrame(res_dict)

    out_csv = f"ffb_traces/{dataset}_{sensitive_attr}_{ml_algo}.csv"
    res_df.to_csv(out_csv)


   


        



if __name__ == "__main__":
    main()