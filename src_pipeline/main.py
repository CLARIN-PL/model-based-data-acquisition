from utils.pipeline import Pipeline
import utils.config as conf
import pandas as pd

pd.options.mode.chained_assignment = None


def pipeline():
    pipe = Pipeline(**conf.PIPELINE_INIT_ARGUMENTS)
    pipe.run(**conf.PIPELINE_RUN_ARGUMENTS)


if __name__ == "__main__":
    pipeline()
