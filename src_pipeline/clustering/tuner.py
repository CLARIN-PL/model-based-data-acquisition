from clustering.clustering_model import ClusteringModel
from utils.config import LABEL_DIMS
import json
import pandas as pd
from utils.text_type import TextType
from tqdm import tqdm


class ClusterTuner:
    def __init__(
        self,
        test_output_file,
        clusters_file="data/cluster_id.csv",
        comments_file="data/comments_cleaned_predicted.csv",
    ) -> None:
        self.test_output_file = test_output_file
        self.clusters_file = clusters_file
        self.comments_file = comments_file

    def set_data(self, data):
        self.data = data

    def find_k_for_entire_data(self, ks=range(2, 20)):
        model = ClusteringModel(self.test_output_file)
        model.set_data(self.data[LABEL_DIMS].drop_duplicates())
        model.test_k(ks)
        return self

    def load_article_ids(self):
        clusters = pd.read_csv(self.clusters_file)
        self.article_ids = clusters.loc[
            clusters.text_type == TextType.ARTICLE.value, "article_id"
        ].unique()
        return self

    def load_comments(self):
        comments = pd.read_csv(self.comments_file)
        self.set_data(comments)
        return self

    def find_average_k(self, min_amount, max_k=10):
        result = []
        model = ClusteringModel(None, verbose=False)
        for art_id in tqdm(self.article_ids):
            curr_data = self.data[self.data.article_id == art_id]
            if curr_data.shape[0] <= min_amount:
                continue
            model.set_data(curr_data[LABEL_DIMS].drop_duplicates())
            samples_amount = model.data.shape[0]
            if samples_amount < 2:
                continue
            ks = range(2, min(max_k + 1, samples_amount))
            model.test_k(ks)
            result.append(model.results)
        with open(self.test_output_file, "w") as f:
            json.dump(result, f)
        return self

    def run_tuning_average_k(self, min_amount, max_k=10):
        return self.load_comments().load_article_ids().find_average_k(min_amount, max_k)
