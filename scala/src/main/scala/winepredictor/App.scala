package winepredictor

import org.apache.spark.sql.SparkSession
import Config.{TRAIN_CSV, VAL_CSV, MODEL_PATH}

/**
 * Main application entry point for Spark Wine Predictor.
 * Orchestrates the entire pipeline.
 */
object App {
  
  def main(args: Array[String]): Unit = {
    val spark = SparkSessionFactory.initializeSpark()
    
    try {
      println("Starting Spark application...")
      
      // Initialize components
      val dataProcessor = new DataProcessor(spark)
      val modelTrainer = new ModelTrainer()
      
      // Load and preprocess data
      println("Loading datasets...")
      var df = dataProcessor.loadDataset(TRAIN_CSV, VAL_CSV)
      
      println("Preprocessing data...")
      df = dataProcessor.preprocessData(df)
      
      println("Oversampling minority classes...")
      df = dataProcessor.oversampleMinorityClasses(df)
      
      println("Splitting data into train and validation sets...")
      val (trainDf, valDf) = dataProcessor.splitData(df)
      
      // Train model
      println("Training Random Forest model with grid search...")
      val bestRfModel = modelTrainer.performGridSearchRF(trainDf, valDf)
      
      // Save model
      println(s"Saving the best model to $MODEL_PATH...")
      bestRfModel.write.overwrite().save(MODEL_PATH)
      
      println("Pipeline completed successfully!")
      
    } catch {
      case e: Exception =>
        println(s"Error during pipeline execution: ${e.getMessage}")
        e.printStackTrace()
        throw e
    } finally {
      println("Stopping Spark session...")
      spark.stop()
    }
  }
}

