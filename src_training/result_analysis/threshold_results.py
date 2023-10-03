import plotly.graph_objects as go
import json
import pandas as pd
from os import path
import numpy as np
import sys

sys.path.insert(0, "/mnt/big_one/swozniak/AnnotClassifier/src/")
from train.dimension_lists_py import label_dims


def get_filepaths(agg_type):
    filename_template = "metrics/crossvalidation_agg{agg_type}_th{threshold}_f10.json"
    thresholds = (
        [x / 100 for x in range(0, 31, 5)] if agg_type == "mean" else range(5, 11)
    )
    return [
        (th, filename_template.format(agg_type=agg_type, threshold=th))
        for th in thresholds
    ]


def get_data(filepath):
    with open(filepath, "r") as f:
        data = json.loads(json.load(f))

    return data


def get_preds_and_true(data):
    def reshape_(array):
        curr_sh = array.shape
        return array.reshape(curr_sh[0] * curr_sh[1], curr_sh[2])

    preds = reshape_(np.array(data["predicted"]))
    true = reshape_(np.array(data["true"]))
    return preds, true


def get_mean_dimension(data):
    avg = data.sum(axis=1).mean()
    lost_indices = np.where(data.sum(axis=0) == 0)[0]
    print(data.shape)
    lost_dimensions = [label_dims[i] for i in lost_indices]
    lost_count = lost_indices.shape[0]
    rounded = round(avg)
    return rounded, avg, lost_count, lost_dimensions


def get_all_thresholds_mean(paths):
    def get_summary(is_preds=True):
        dict_ = {
            "threshold": [],
            "rounded": [],
            "average": [],
            "lost_count": [],
            "lost_dimensions": [],
        }
        for th, p in paths:
            data = get_data(p)
            preds, true = get_preds_and_true(data)
            rounded, avg, lost_count, lost_dimensions = get_mean_dimension(
                preds if is_preds else true
            )
            dict_["threshold"].append(th)
            dict_["rounded"].append(rounded)
            dict_["average"].append(avg)
            dict_["lost_count"].append(lost_count)
            dict_["lost_dimensions"].append(lost_dimensions)

        return pd.DataFrame(data=dict_)

    summary_preds = get_summary(is_preds=True)
    summary_true = get_summary(is_preds=False)
    return summary_preds, summary_true


if __name__ == "__main__":
    paths = get_filepaths("mean")
    data_pred, data_true = get_all_thresholds_mean(paths)
    print(data_pred)
    print(data_true)
