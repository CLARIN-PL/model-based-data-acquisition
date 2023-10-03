from utils.config import LABEL_DIMS
from clustering.clustering_model import ClusteringModel
import pandas as pd


class ClusterizerInsideArticle:
    def __init__(self, data, N, max_k, sampling_method="balanced"):
        self.data = data
        self.N = N
        self.max_k = max_k
        assert sampling_method in ["balanced", "most_numerous"]
        self.sampling_method = sampling_method

    def sampling(self):
        if self.sampling_method == "balanced":
            return self.balanced_sampling()
        if self.sampling_method == "most_numerous":
            return self.most_numerous_sampling()
        return None

    def balanced_sampling(self):
        curr_cluster_id = 0
        no_clusters = self.data["cluster_id"].max() + 1
        sampled = pd.DataFrame(columns=self.data.columns)
        for _ in range(self.N):
            curr_cluster = self.data[self.data["cluster_id"] == curr_cluster_id]
            while curr_cluster.shape[0] == 0:
                curr_cluster_id += 1
                curr_cluster_id %= no_clusters
                curr_cluster = self.data[self.data["cluster_id"] == curr_cluster_id]
            curr_cluster_id += 1
            curr_cluster_id %= no_clusters
            sample = curr_cluster.sample(1)
            self.data = pd.concat([self.data, sample], axis=0).drop_duplicates(
                subset=["article_id", "id_inside_article", "cluster_id"] + LABEL_DIMS,
                keep=False,
            )
            sampled = pd.concat([sampled, sample], axis=0)
        return sampled

    def most_numerous_sampling(self):
        sorted = self.data["cluster_id"].value_counts().reset_index()
        # print(sorted.columns)
        # print(sorted.iloc[:, 0])
        bests = sorted[sorted["count"] == sorted["count"].max()]["cluster_id"]
        sampled_cluster_id = bests.sample(1).item()
        return self.data[self.data["cluster_id"] == sampled_cluster_id]

    def clusterize(self):
        train_data = self.data[LABEL_DIMS].drop_duplicates()
        if train_data.shape[0] <= 1:
            return self.data
        curr_k = min(self.max_k, train_data.shape[0])
        clustering_model = ClusteringModel(k=curr_k)
        clustering_model.set_data(train_data)
        clustering_model.train()
        self.data["cluster_id"] = clustering_model.model.predict(self.data[LABEL_DIMS])

        #! --------------
        # self.data["vector"] = self.data[LABEL_DIMS].apply(
        #     (lambda x: str(x.tolist())),
        #     axis=1
        # )
        # # print(clustering_model.model.cluster_centers_)
        # self.data["centroid"] = self.data["cluster_id"].apply(
        #     (lambda x: clustering_model.model.cluster_centers_[x]),
        # )
        #!

    def run(self):
        # self.data["vector"] = 0
        # self.data["centroid"] = 0
        # self.data["cluster_id"] = -1

        if self.data.shape[0] <= self.N:
            return self.data
        self.clusterize()
        sampled = self.sampling()
        sampled.drop(columns=["cluster_id"], inplace=True)
        return sampled
