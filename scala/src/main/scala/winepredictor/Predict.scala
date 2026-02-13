package winepredictor

import org.apache.spark.ml.classification.{LogisticRegressionModel, RandomForestClassificationModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.SparkSession
import Config.MODEL_PATH

import java.io.File
import scala.io.Source
import scala.util.Using

/**
 * Prediction script for Spark Wine Predictor.
 * Loads a trained model and makes predictions on test data.
 */
object Predict {
  
  private val TEST_CSV = "TestDataset.csv"
  
  def main(args: Array[String]): Unit = {
    val spark = SparkSessionFactory.initializeSpark()
    
    try {
      // Initialize data processor
      val dataProcessor = new DataProcessor(spark)
      
      // Load test dataset
      println("Loading test dataset...")
      var testDf = spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .option("sep", ";")
        .csv(TEST_CSV)
      
      testDf = dataProcessor.preprocessData(testDf)
      
      // Load saved model
      println("Loading the trained model...")
      val bestModel = loadModel(MODEL_PATH)
      
      // Make predictions
      println("Making predictions...")
      val preds = bestModel.transform(testDf)
      preds.select("features", "prediction").show()
      
      // Evaluate model
      println("Evaluating model performance...")
      val (accuracy, f1Score) = evaluateModel(preds)
      println(s"Model Performance: \n Accuracy = $accuracy%.4f \n F1 Score = $f1Score%.4f")
      
    } catch {
      case e: Exception =>
        println(s"An error occurred: ${e.getMessage}")
        e.printStackTrace()
    } finally {
      println("Stopping Spark session...")
      spark.stop()
    }
  }
  
  private def loadModel(modelPath: String): org.apache.spark.ml.Model[_] = {
    val modelType = getModelType(modelPath)
    
    modelType match {
      case "LogisticRegressionModel" =>
        LogisticRegressionModel.load(modelPath)
      case "RandomForestClassificationModel" =>
        RandomForestClassificationModel.load(modelPath)
      case _ =>
        throw new IllegalArgumentException(s"Unsupported model type: $modelType")
    }
  }
  
  private def getModelType(modelPath: String): String = {
    val metadataPath = s"$modelPath/metadata/part-00000"
    val file = new File(metadataPath)
    
    if (!file.exists()) {
      throw new FileNotFoundException(s"Model metadata not found at: $metadataPath")
    }
    
    Using(Source.fromFile(file)) { source =>
      val content = source.mkString
      // Simple JSON parsing - extract class name
      val classPattern = """"class"\s*:\s*"([^"]+)"""".r
      classPattern.findFirstMatchIn(content) match {
        case Some(m) =>
          val className = m.group(1)
          className.split("\\.").last
        case None =>
          throw new IllegalArgumentException("Could not parse model class from metadata")
      }
    }.get
  }
  
  private def evaluateModel(predictions: org.apache.spark.sql.DataFrame): (Double, Double) = {
    val accEval = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    
    val f1sEval = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("f1")
    
    val accuracy = accEval.evaluate(predictions)
    val f1Score = f1sEval.evaluate(predictions)
    
    (accuracy, f1Score)
  }
}

