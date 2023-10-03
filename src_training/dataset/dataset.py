from torch.utils.data import Dataset
import torch
from typing import List, Tuple, Optional
import pandas as pd


class AnnotatedDataset(Dataset):
    LABEL_COLUMNS = [
        "Pozytywne",
        "Negatywne",
        "Radość",
        "Zachwyt",
        "Inspiruje",
        "Spokój",
        "Zaskoczenie",
        "Współczucie",
        "Strach",
        "Smutek",
        "Wstręt",
        "Złość",
        "Ironiczny",
        "Żenujący",
        "Wulgarny",
        "Polityczny",
        "Interesujący",
        "Zrozumiały",
        "Potrzebuję więcej informacji, aby ocenić ten tekst",
        "Obraża mnie",
        "Może kogoś atakować / obrażać / lekceważyć",
        "Mnie bawi/śmieszy?",
        "Może kogoś bawić?",
    ]

    def __init__(
        self, data: pd.DataFrame, label_columns: Optional[List[str]] = None
    ) -> None:
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
        labels = curr[self.label_columns].tolist()
        return text, torch.tensor(labels, dtype=torch.float16)
