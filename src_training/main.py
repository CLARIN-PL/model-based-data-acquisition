from dataset.datamodule import AnnotatedDataModule, AnnotatedDataset
from experiments.unhealthy import UnhealthyDataModule, UnhealthyDataset
from experiments.hate_speech import HateSpeechDataModule, HateSpeechDataset
from train.crossvalidation import cross_validation
from train.single import single_training
from tqdm import tqdm
from experiments.crossvalidation import CrossValidation


def run_CV(datamodule, dataset, transformer_name, name, enough_time):
    cv = CrossValidation(k_folds=10,
                         datamodule_class=datamodule,
                         dataset_class=dataset,
                         transformer_name=transformer_name,
                         name=name)
    # cv.run_A_B(run_A=True, run_B=True)
    # cv.run_increment()
    # if enough_time:
    # cv.run_threshold()
    cv.run_single()

def test_hate_speech_datamodule():
    datamodule = HateSpeechDataModule(preprocessing_args=(0.15, "mean"))

    datamodule.prepare_data()
    datamodule.setup()
    
def test_unhealthy_datamodule():
    datamodule = UnhealthyDataModule(preprocessing_args=(0.15, "mean"))

    datamodule.prepare_data()
    datamodule.setup()
    
if __name__ == "__main__":
    # test_unhealthy_datamodule()
    # test_hate_speech_datamodule()
    
    # run_CV(datamodule=AnnotatedDataModule, dataset=AnnotatedDataset, transformer_name="allegro/herbert-base-cased", name="doccano1", enough_time=True)
    run_CV(datamodule=HateSpeechDataModule, dataset=HateSpeechDataset, transformer_name="xlm-roberta-base", name="hate_speech", enough_time=True)
    # run_CV(datamodule=UnhealthyDataModule, dataset=UnhealthyDataset, transformer_name="xlm-roberta-base", name="unhealthy", enough_time=True)
    
