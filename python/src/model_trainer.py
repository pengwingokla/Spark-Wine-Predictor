"""
Model training and evaluation module.
This module handles model training, hyperparameter tuning, and evaluation.
"""
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from .config import MODEL_CONFIG


class ModelTrainer:
    """Handles model training and evaluation."""
    
    def __init__(self):
        """Initialize ModelTrainer."""
        self.accuracy_evaluator = MulticlassClassificationEvaluator(
            labelCol="label", 
            predictionCol="prediction", 
            metricName="accuracy"
        )
        self.f1_evaluator = MulticlassClassificationEvaluator(
            labelCol="label", 
            predictionCol="prediction", 
            metricName="f1"
        )
    
    def perform_grid_search_rf(self, train_df, val_df):
        """
        Perform grid search for Random Forest hyperparameter tuning.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            
        Returns:
            RandomForestModel: Best model from grid search
        """
        rf = RandomForestClassifier(featuresCol="features", labelCol="label")
        
        paramGrid = ParamGridBuilder() \
            .addGrid(rf.numTrees, MODEL_CONFIG["num_trees"]) \
            .addGrid(rf.maxDepth, MODEL_CONFIG["max_depth"]) \
            .addGrid(rf.maxBins, MODEL_CONFIG["max_bins"]) \
            .build()
        
        tvs = TrainValidationSplit(
            estimator=rf,
            estimatorParamMaps=paramGrid,
            evaluator=self.accuracy_evaluator,
            trainRatio=MODEL_CONFIG["train_validation_split"]
        )
        
        model = tvs.fit(train_df)
        best_rf = model.bestModel
        
        val_preds = best_rf.transform(val_df)
        best_accuracy = self.accuracy_evaluator.evaluate(val_preds)
        best_f1_score = self.f1_evaluator.evaluate(val_preds)
        
        print("\nBest Random Forest Model:")
        print(f"numTrees = {best_rf.getNumTrees}")
        print(f"maxDepth = {best_rf.getMaxDepth}")
        print(f"maxBins  = {best_rf.getMaxBins}")
        print()
        print(f"BEST ACCURACY ACHIEVED: {best_accuracy:.4f}")
        print(f"BEST F1 SCORE ACHIEVED: {best_f1_score:.4f}")
        print()
        
        return best_rf
    
    def train_and_evaluate(self, models, train_df, val_df):
        """
        Train multiple models and evaluate them.
        
        Args:
            models: List of tuples (name, model_instance)
            train_df: Training DataFrame
            val_df: Validation DataFrame
            
        Returns:
            list: List of tuples (name, accuracy, f1_score)
        """
        results = []
        
        for name, model in models:
            print(f"\nTraining {name}...")
            spark_model = model.fit(train_df)
            preds = spark_model.transform(val_df)
            accur = self.accuracy_evaluator.evaluate(preds)
            f1_score = self.f1_evaluator.evaluate(preds)
            print(f"Accuracy: {accur:.4f}, \nF1 Score: {f1_score:.4f}")
            results.append((name, accur, f1_score))
        
        return results

