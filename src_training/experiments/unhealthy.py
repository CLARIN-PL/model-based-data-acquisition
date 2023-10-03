from torch.utils.data import DataLoader
import pytorch_lightning as pl
import pandas as pd
from dataset.preprocessing import preprocessing
from os.path import join
from typing import List, Optional, Tuple
from torch.utils.data import Dataset
import torch


class UnhealthyDataset(Dataset):
    LABEL_COLUMNS = [
        "antagonize",
        "condescending",
        "dismissive",
        "generalisation_unfair",
        "hostile",
        "sarcastic"
    ]
    
    def __init__(self, data: pd.DataFrame, label_columns: Optional[List[str]] = None) -> None:
        super().__init__()
        self.data = data
        self.label_columns = (
            self.LABEL_COLUMNS if label_columns is None else label_columns
        )
        
    def __len__(self) -> int:
        return self.data.shape[0]
    
    def __getitem__(self, index: int) -> Tuple[str, List[int]]:
        curr = self.data.iloc[index]
        text = curr["text"]
        labels = curr[self.label_columns]
        # if len(self.label_columns) > 1:
        #     labels = torch.tensor(labels.tolist(), dtype=torch.float16)
        # else:
        #     labels = labels.item()
        return text, torch.tensor(labels.tolist(), dtype=torch.float16)
    
    
class UnhealthyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 16,
        folds: int = 10,
        datadir: str = "data/train",
        preprocessing_args: Tuple[float, str] = (1.0, "mean"),
        label_columns: Optional[List[str]] = None,
        dev_fold: int = 0,
    ) -> None:
        super().__init__()
        self.datadir = datadir
        self.batch_size = batch_size
        self.folds_num = folds
        self.preprocessing_args = preprocessing_args
        # name_id = "_".join(self.preprocessing_args)
        self.data_name = f"unhelathy_data_{self.preprocessing_args[1]}_{self.preprocessing_args[0]}_{self.folds_num}.csv"
        self.label_columns = (
            UnhealthyDataset.LABEL_COLUMNS if label_columns is None else label_columns
        )
        self.dev_fold = dev_fold
    
    def set_txt_id(self) -> None:
        filepath = join(self.datadir, "unhealthy.csv")
        data = pd.read_csv(filepath)
        texts = data[["comment"]].drop_duplicates()
        texts["text_id"] = [i for i in range(texts.shape[0])]
        data = pd.merge(data, texts, how="inner", on="comment")
        data.rename(columns={"comment": "text"}, inplace=True)
        data.to_csv(join(self.datadir, "unhealthy_with_id.csv"), index=False)
        return data
    
    def prepare_data(self) -> None:
        self.set_txt_id()
        data = preprocessing(
            datadir=self.datadir,
            data_file="unhealthy_with_id.csv",
            annot_file="unhealthy_with_id.csv",
            aggregation_type=self.preprocessing_args[1],
            threshold=self.preprocessing_args[0],
            label_columns=self.label_columns
        )
        data = data.dropna()
        folded = self.create_folds(data)

        filepath = join(self.datadir, self.data_name)
        folded.to_csv(filepath, header=True, index=False)
        
    def setup(self, stage: str = "fit") -> None:
        filepath = join(self.datadir, self.data_name)
        self.data = pd.read_csv(filepath)
        test_fold = (self.dev_fold + 1) % self.folds_num
        self.data["split"] = "train"
        if self.dev_fold == -1:
            self.data.loc[self.data["fold"] == test_fold, "split"] = "dev"
        else:
            self.data.loc[self.data["fold"] == test_fold, "split"] = "test"
            self.data.loc[self.data["fold"] == self.dev_fold, "split"] = "dev"

    def create_folds(self, data: pd.DataFrame) -> pd.DataFrame:
        sort = data["text_id"].value_counts().sort_values(ascending=False)

        data["fold"] = -1
        for num, (text_id, _) in enumerate(sort.items()):
            curr_fold = num % self.folds_num
            data.loc[data["text_id"] == text_id, "fold"] = curr_fold
        return data

    def get_split(self, split: str) -> UnhealthyDataset:
        splitted = self.data[self.data["split"] == split]
        dataset = UnhealthyDataset(splitted, self.label_columns)
        return dataset

    def train_dataloader(self) -> DataLoader:
        split_dataset = self.get_split("train")
        return DataLoader(split_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        split_dataset = self.get_split("dev")
        return DataLoader(split_dataset, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        split_dataset = self.get_split("test")
        return DataLoader(split_dataset, batch_size=self.batch_size)