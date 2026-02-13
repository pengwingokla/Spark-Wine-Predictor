"""
Spark session initialization module.
This module handles Spark session creation and configuration.
"""
from pyspark.sql import SparkSession
from .config import SPARK_CONFIG


def initialize_spark():
    """
    Initialize and return a SparkSession with configured settings.
    
    Returns:
        SparkSession: Configured Spark session
    """
    return SparkSession.builder \
        .appName(SPARK_CONFIG["app_name"]) \
        .master(SPARK_CONFIG["master"]) \
        .config("spark.executor.memory", SPARK_CONFIG["executor_memory"]) \
        .config("spark.executor.cores", SPARK_CONFIG["executor_cores"]) \
        .config("spark.task.cpus", SPARK_CONFIG["task_cpus"]) \
        .getOrCreate()

