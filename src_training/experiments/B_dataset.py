from torch.utils.data import Dataset
import torch
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np


class BDataset(Dataset):
    def __init__(self, data: pd.DataFrame, state: str) -> None:
        super().__init__()
        self.data = data
        self.A_label_column = "originals"
        self.B_label_column = "predicted"
        self.trainer_state = state

    def train(self):
        self.trainer_state = "train"

    def test(self):
        self.trainer_state = "test"

    def string_to_tensor(self, labels):
        clean = np.fromstring(
            str(labels).replace(",", "").replace("[", "").replace("]", ""), sep=" "
        )
        return torch.tensor(clean, dtype=torch.float16)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int) -> Tuple[str, List[int]]:
        curr = self.data.iloc[index]
        text = curr["text"]
        labels = self.string_to_tensor(curr[self.B_label_column])
        return text, labels
