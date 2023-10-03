import pandas as pd
from typing import List, Optional
from os.path import join
import numpy as np


def major_vote(
    df: pd.DataFrame,
    label_columns: List[str],
    threshold: float = 0.0,
    aggregation_type: str = "sum",
) -> pd.DataFrame:
    means = df[label_columns + ["text_id"]].groupby(["text_id"])
    if aggregation_type == "sum":
        means = means.sum().reset_index()
    else:
        means = means.mean().reset_index()
    condition = means[label_columns] >= threshold

    means[label_columns] = np.where(condition, 1, 0)
    return means


def binarization(df: pd.DataFrame, label_columns: List[str]) -> pd.DataFrame:
    df[label_columns] = np.where(df[label_columns] > 0, 1, 0)
    return df


def preprocessing(
    datadir: str = "data/train",
    data_file: str = "data.csv",
    annot_file: str = "annotations.csv",
    threshold: float = 1.0,
    aggregation_type: str = "sum",
    label_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    annot_file = join(datadir, annot_file)        
    df = pd.read_csv(annot_file)

    if label_columns is None:
        label_columns = list(set(df.columns) - {"text_id", "user_id"})

    df = binarization(df, label_columns)
    df = major_vote(
        df,
        label_columns=label_columns,
        threshold=threshold,
        aggregation_type=aggregation_type,
    )

    text_file = join(datadir, data_file)
    df_txt = pd.read_csv(text_file)
    df = df.merge(df_txt[["text_id", "text"]], on="text_id").drop_duplicates()
    # df.drop(columns=drop_columns, inplace=True)

    return df
