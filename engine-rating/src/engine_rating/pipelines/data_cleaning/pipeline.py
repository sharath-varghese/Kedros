
from kedro.pipeline import Pipeline, node
from .nodes import cleaning

def create_pipeline(**kwargs):
    return Pipeline(
            [
                node(func=cleaning,inputs='raw_data',outputs='clean_data',name='data_cleaning'),
            ])
