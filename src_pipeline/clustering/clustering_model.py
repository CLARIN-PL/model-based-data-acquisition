from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
import json
from tqdm import tqdm


class ClusteringModel:
    def __init__(self, test_output_file=None, k=2, verbose=True):
        self.test_output_file = test_output_file
        self.k = k
        self.verbose = verbose

    def set_data(self, data):
        self.data = data

    def set_k(self, k):
        if k < 2:
            raise ValueError("K too low")
        self.k = k

    def train(self):
        self.model = MiniBatchKMeans(n_clusters=self.k)
        self.model.fit(self.data)
        return self

    def information_criteria(self):
        gmm = GaussianMixture(n_components=self.k, init_params="kmeans")
        gmm.fit(self.data)
        self.aic = gmm.aic(self.data)
        self.bic = gmm.bic(self.data)
        return self

    def test_k(self, ks=range(2, 20)):
        self.results = {}
        iterator = tqdm(ks) if self.verbose else ks
        for k in iterator:
            self.set_k(k)
            self.train()
            score = silhouette_score(self.data, self.model.labels_)
            iner = self.model.inertia_
            self.information_criteria()
            self.results[k] = [score, iner, self.aic, self.bic]
        if self.test_output_file is not None:
            with open(self.test_output_file, "w") as f:
                json.dump(json.dumps(self.results), f, indent=4)
        return self
