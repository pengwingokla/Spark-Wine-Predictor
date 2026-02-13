"""
Configuration module for Spark Wine Predictor.
This module centralizes all configuration settings.
"""
import os

# Dataset paths
TRAINCSV = os.getenv("TRAIN_CSV", "TrainingDataset.csv")
VALCSV = os.getenv("VAL_CSV", "ValidationDataset.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "best_model")

# Spark configuration
SPARK_CONFIG = {
    "app_name": "WineQualityPrediction",
    "master": os.getenv("SPARK_MASTER", "spark://172.31.87.32:7077"),
    # Use "local[*]" for local development
    "executor_memory": os.getenv("SPARK_EXECUTOR_MEMORY", "3g"),
    "executor_cores": os.getenv("SPARK_EXECUTOR_CORES", "2"),
    "task_cpus": os.getenv("SPARK_TASK_CPUS", "1"),
}

# Data processing configuration
DATA_CONFIG = {
    "csv_separator": ";",
    "train_ratio": 0.8,
    "random_seed": 42,
}

# Model training configuration
MODEL_CONFIG = {
    "num_trees": [20, 50, 75, 100, 150, 200],
    "max_depth": [5, 10, 15, 20, 25],
    "max_bins": [16, 32, 48, 64],
    "train_validation_split": 0.8,
}

