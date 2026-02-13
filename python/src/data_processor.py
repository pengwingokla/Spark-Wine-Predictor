"""
Data processing module.
This module handles data loading, preprocessing, and feature engineering.
"""
import os
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, count
from pyspark.sql.types import NumericType
from .config import DATA_CONFIG


class DataProcessor:
    """Handles all data processing operations."""
    
    def __init__(self, spark):
        """
        Initialize DataProcessor with Spark session.
        
        Args:
            spark: SparkSession instance
        """
        self.spark = spark
    
    def load_dataset(self, train_csv, val_csv):
        """
        Load training and validation datasets and combine them.
        
        Args:
            train_csv: Path to training CSV file
            val_csv: Path to validation CSV file
            
        Returns:
            DataFrame: Combined dataset
        """
        if not os.path.exists(train_csv) or not os.path.exists(val_csv):
            raise FileNotFoundError("The dataset files do not exist.")
        
        train_df = self.spark.read.csv(
            train_csv, 
            header=True, 
            inferSchema=True, 
            sep=DATA_CONFIG["csv_separator"]
        )
        val_df = self.spark.read.csv(
            val_csv, 
            header=True, 
            inferSchema=True, 
            sep=DATA_CONFIG["csv_separator"]
        )
        
        return train_df.union(val_df)
    
    def preprocess_data(self, df):
        """
        Preprocess the dataset: clean column names, create features, and label.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame: Preprocessed DataFrame with 'features' and 'label' columns
        """
        # Remove extra quotes from column names
        df = df.toDF(*[c.replace('"', '') for c in df.columns])
        
        # Features: Exclude the 'quality' column
        feature_cols = df.columns[:-1]
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        df = assembler.transform(df)
        
        # Target: Ensure 'quality' column is numeric
        if isinstance(df.schema['quality'].dataType, NumericType):
            # If 'quality' is numeric, rename it to 'label'
            df = df.withColumnRenamed('quality', 'label')
        else:
            # If 'quality' is not numeric, index it
            indexer = StringIndexer(inputCol="quality", outputCol="label")
            df = indexer.fit(df).transform(df)
        
        return df
    
    def oversample_minority_classes(self, df):
        """
        Oversample minority classes to balance the dataset.
        
        Args:
            df: Input DataFrame with 'label' column
            
        Returns:
            DataFrame: Oversampled DataFrame
        """
        # Check class distribution
        tot_count = df.count()
        
        print("Original class distribution:")
        cls_dist = df.groupBy("label") \
            .agg(count("label").alias("cls_count")) \
            .withColumn("percentage", (col("cls_count") / tot_count) * 100) \
            .orderBy("label")
        cls_dist.show()
        
        # Maximum class count
        max_count = cls_dist.agg({"cls_count": "max"}).collect()[0][0]
        
        # Oversample minority classes
        oversampled_df = df
        for row in cls_dist.collect():
            label = row["label"]
            cls_count = row["cls_count"]
            if cls_count < max_count:
                fraction = (max_count - cls_count) / cls_count
                sampled_df = df.filter(col("label") == label).sample(
                    withReplacement=True, 
                    fraction=fraction, 
                    seed=DATA_CONFIG["random_seed"]
                )
                oversampled_df = oversampled_df.union(sampled_df)
        
        # Verify new class distribution
        print("New class distribution:")
        new_dist = oversampled_df.groupBy("label") \
            .agg(count("label").alias("cls_count")) \
            .withColumn("percentage", (col("cls_count") / tot_count) * 100) \
            .orderBy("label")
        new_dist.show()
        
        return oversampled_df
    
    def split_data(self, df):
        """
        Split dataset into training and validation sets.
        
        Args:
            df: Input DataFrame
            
        Returns:
            tuple: (train_df, val_df)
        """
        return df.randomSplit(
            [DATA_CONFIG["train_ratio"], 1 - DATA_CONFIG["train_ratio"]], 
            seed=DATA_CONFIG["random_seed"]
        )

