from cleaning.cleaner import ParagraphCleaner, CommentCleaner
from utils.reader import TextReader
from predicting.predictor import TextPredictor
from clustering.clusterizer import TextClusterizer
from sampling.sampler import TextSampler
from clustering.tuner import ClusterTuner
import time


class Pipeline:
    def __init__(
        self,
        reader_args={},
        paragraph_cleaner_args={},
        comment_cleaner_args={},
        paragraph_predictor_args={},
        comment_predictor_args={},
        clusterizer_args={},
        comment_cluster_tuner_args={},
        sampler_args={},
    ):
        self.reader = TextReader(**reader_args)

        self.comment_cleaner = CommentCleaner(**comment_cleaner_args)
        self.paragraph_cleaner = ParagraphCleaner(**paragraph_cleaner_args)

        self.comment_predictor = TextPredictor(**comment_predictor_args)
        self.paragraph_predictor = TextPredictor(**paragraph_predictor_args)

        self.clusterizer = TextClusterizer(**clusterizer_args)
        self.comment_cluster_tuner = ClusterTuner(**comment_cluster_tuner_args)
        self.sampler = TextSampler(**sampler_args)

    def _single_run(self, runner_func, name, runner_args={}):
        if self.verbose:
            print(f"{name} running")
        start = time.time()
        runner_func(**runner_args)
        end = time.time()
        if self.verbose:
            print(f"{name} finished in {end-start} seconds")

    def run(
        self,
        reader=False,
        paragraph_cleaner=False,
        comment_cleaner=False,
        paragraph_predictor=False,
        comment_predictor=False,
        clusterizer_prediction=False,
        clusterizer_tuning=False,
        clusterizer_tuning_args={},
        comment_cluster_tuning=False,
        comment_cluster_tuning_args={},
        sampler=False,
        sampler_args={},
        all_true=True,
        verbose=True,
    ):
        self.verbose = verbose
        start = time.time()
        if reader or all_true:
            self._single_run(self.reader.run, "Reader")
        if paragraph_cleaner or all_true:
            self._single_run(self.paragraph_cleaner.run, "Paragraph Cleaner")
        if comment_cleaner or all_true:
            self._single_run(self.comment_cleaner.run, "Comment Cleaner")
        if paragraph_predictor or all_true:
            self._single_run(self.paragraph_predictor.run, "Paragraph Predictor")
        if comment_predictor or all_true:
            self._single_run(self.comment_predictor.run, "Comment Predictor")
        if clusterizer_tuning:
            self._single_run(
                self.clusterizer.run_tuning_k,
                "Clusterizer tuning k",
                clusterizer_tuning_args,
            )
        if clusterizer_prediction or all_true:
            self._single_run(self.clusterizer.run_prediction, "Clusterizer prediction")
        if comment_cluster_tuning:
            self._single_run(
                self.comment_cluster_tuner.run_tuning_average_k,
                "Comment clusterizing tuning k",
                comment_cluster_tuning_args,
            )
        if sampler or all_true:
            self._single_run(self.sampler.run, "Sampler", sampler_args)
        end = time.time()
        if self.verbose:
            print(f"Pipeline finished in {end-start} seconds")
        return self
