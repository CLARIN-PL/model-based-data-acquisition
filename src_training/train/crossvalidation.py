from model.model import ClassificationModel
from dataset.datamodule import AnnotatedDataModule
import pytorch_lightning as pl
import numpy as np
from copy import copy
import json
from os import path
from train.dimension_lists_py import dimension_splits, label_dims


def get_model(out_size):
    return ClassificationModel(out_size=out_size)


def get_trainer():
    callback_es = pl.callbacks.EarlyStopping(monitor="val_f1_macro", patience=3)

    trainer = pl.Trainer(
        enable_checkpointing=False,
        accelerator="auto",
        devices=1,
        auto_select_gpus=True,
        max_epochs=10,
        callbacks=[callback_es],
    )
    return trainer


def one_iteration(iter, preprocessing_args, folds, label_columns):
    datamodule = AnnotatedDataModule(
        folds=folds, preprocessing_args=preprocessing_args, label_columns=label_columns
    )
    if iter == 0:
        datamodule.prepare_data()
    datamodule.setup(dev_fold=iter)

    model = get_model(len(label_columns))
    trainer = get_trainer()

    trainer.fit(model=model, datamodule=datamodule)

    trainer.test(model=model, datamodule=datamodule)

    return model.last_metrics


def loop(folds=10, aggregation_type="sum", label_columns=[]):
    if aggregation_type == "sum":
        thresholds = range(5, 11)
    else:
        thresholds = [th / 100 for th in range(0, 31, 5)]

    print(thresholds)
    for threshold in thresholds:
        metrics_results = []
        preprocessing_args = (threshold, aggregation_type)
        name = f"agg{aggregation_type}_th{threshold}"
        for i in range(folds):
            metrics = one_iteration(i, preprocessing_args, folds, label_columns)
            metrics_results.append(metrics)
        metrics_by_name = aggregate_results(metrics_results)
        full_metrics = calculate_distribution(metrics_by_name)
        fname = f"crossvalidation_{name}_f{folds}.json"
        save_metrics(full_metrics, fname)
    return None


def aggregate_results(results):
    aggregated = {}
    for res in results:
        for k, v in res.items():
            if k not in aggregated:
                aggregated[k] = []
            if "class" in k:
                value = v.tolist()
            elif k == "predicted" or k == "true":
                value = v
            else:
                value = v.item()
            aggregated[k].append(value)
    return aggregated


def calculate_distribution(results):
    copy_results = copy(results)
    for k, v in copy_results.items():
        if "class" in k:
            results[k] = v
            # if len(np.array(v).shape) > 1:
            results[f"{k}_mean"] = list(np.mean(v, axis=0))
            results[f"{k}_std"] = list(np.std(v, axis=0))
        elif k == "predicted" or k == "true":
            results[k] = v
        else:
            results[f"{k}_mean"] = np.mean(v)
            results[f"{k}_std"] = np.std(v)
    return results


def cross_validation(folds=10, aggregation_type="sum"):
    loop(folds=folds, aggregation_type=aggregation_type, label_columns=label_dims)
