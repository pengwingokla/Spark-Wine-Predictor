"""
Main application entry point for Spark Wine Predictor.
This module orchestrates the entire pipeline.
"""
import logging
from src import initialize_spark, DataProcessor, ModelTrainer, TRAINCSV, VALCSV, MODEL_PATH

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WineQualityPrediction")


def main():
    """
    Main execution function that orchestrates the entire pipeline.
    """
    spark = initialize_spark()
    try:
        logger.info("Starting Spark application...")
        
        # Initialize components
        data_processor = DataProcessor(spark)
        model_trainer = ModelTrainer()
        
        # Load and preprocess data
        logger.info("Loading datasets...")
        df = data_processor.load_dataset(TRAINCSV, VALCSV)
        
        logger.info("Preprocessing data...")
        df = data_processor.preprocess_data(df)
        
        logger.info("Oversampling minority classes...")
        df = data_processor.oversample_minority_classes(df)
        
        logger.info("Splitting data into train and validation sets...")
        train_df, val_df = data_processor.split_data(df)
        
        # Train model
        logger.info("Training Random Forest model with grid search...")
        best_rf_model = model_trainer.perform_grid_search_rf(train_df, val_df)
        
        # Save model
        logger.info(f"Saving the best model to {MODEL_PATH}...")
        best_rf_model.write().overwrite().save(MODEL_PATH)
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during pipeline execution: {e}", exc_info=True)
        raise
    finally:
        logger.info("Stopping Spark session...")
        spark.stop()

if __name__ == "__main__":
    main()

