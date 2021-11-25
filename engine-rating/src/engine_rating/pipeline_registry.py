
from typing import Dict

from kedro.pipeline import Pipeline
from engine_rating.pipelines import data_cleaning,model_eda,feature_engineering,model_training
from engine_rating.pipelines.data_cleaning.pipeline import create_pipeline

def register_pipelines() -> Dict[str, Pipeline]:
    dc_pipeline = data_cleaning.create_pipeline()
    eda_pipeline = model_eda.create_pipeline()
    fe_pipeline = feature_engineering.create_pipeline()
    model_pipeline = model_training.create_pipeline()
    return {
            "__default__":dc_pipeline + eda_pipeline + fe_pipeline + model_pipeline,
            "dc":dc_pipeline,
            "eda":eda_pipeline,
            "fe":fe_pipeline,
            "training":model_pipeline
            }

