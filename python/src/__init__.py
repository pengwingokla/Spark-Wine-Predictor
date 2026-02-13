"""
Wine Predictor Package
A Spark-based machine learning pipeline for wine quality prediction.
"""

__version__ = "1.0.0"

from .config import TRAINCSV, VALCSV, MODEL_PATH, SPARK_CONFIG, DATA_CONFIG, MODEL_CONFIG
from .spark_session import initialize_spark
from .data_processor import DataProcessor
from .model_trainer import ModelTrainer

__all__ = [
    "TRAINCSV",
    "VALCSV",
    "MODEL_PATH",
    "SPARK_CONFIG",
    "DATA_CONFIG",
    "MODEL_CONFIG",
    "initialize_spark",
    "DataProcessor",
    "ModelTrainer",
]

