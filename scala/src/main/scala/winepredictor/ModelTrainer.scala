package winepredictor

import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.DataFrame
import Config.ModelConfig

/**
 * Model training and evaluation class.
 * Handles model training, hyperparameter tuning, and evaluation.
 */
class ModelTrainer {
  
  private val accuracyEvaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("label")
    .setPredictionCol("prediction")
    .setMetricName("accuracy")
  
  private val f1Evaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("label")
    .setPredictionCol("prediction")
    .setMetricName("f1")

  /**
   * Perform grid search for Random Forest hyperparameter tuning.
   * 
   * @param trainDf Training DataFrame
   * @param valDf Validation DataFrame
   * @return Best RandomForest model from grid search
   */
  def performGridSearchRF(trainDf: DataFrame, valDf: DataFrame): org.apache.spark.ml.classification.RandomForestClassificationModel = {
    val rf = new RandomForestClassifier()
      .setFeaturesCol("features")
      .setLabelCol("label")
    
    val paramGrid = new ParamGridBuilder()
      .addGrid(rf.numTrees, ModelConfig.NUM_TREES)
      .addGrid(rf.maxDepth, ModelConfig.MAX_DEPTH)
      .addGrid(rf.maxBins, ModelConfig.MAX_BINS)
      .build()
    
    val tvs = new TrainValidationSplit()
      .setEstimator(rf)
      .setEvaluator(accuracyEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(ModelConfig.TRAIN_VALIDATION_SPLIT)
    
    val model = tvs.fit(trainDf)
    val bestRf = model.bestModel.asInstanceOf[org.apache.spark.ml.classification.RandomForestClassificationModel]
    
    val valPreds = bestRf.transform(valDf)
    val bestAccuracy = accuracyEvaluator.evaluate(valPreds)
    val bestF1Score = f1Evaluator.evaluate(valPreds)
    
    println("\nBest Random Forest Model:")
    println(s"numTrees = ${bestRf.getNumTrees}")
    println(s"maxDepth = ${bestRf.getMaxDepth}")
    println(s"maxBins  = ${bestRf.getMaxBins}")
    println()
    println(f"BEST ACCURACY ACHIEVED: $bestAccuracy%.4f")
    println(f"BEST F1 SCORE ACHIEVED: $bestF1Score%.4f")
    println()
    
    bestRf
  }

  /**
   * Train multiple models and evaluate them.
   * 
   * @param models List of tuples (name, model_instance)
   * @param trainDf Training DataFrame
   * @param valDf Validation DataFrame
   * @return List of tuples (name, accuracy, f1_score)
   */
  def trainAndEvaluate(
    models: Seq[(String, org.apache.spark.ml.Predictor[_, _, _])],
    trainDf: DataFrame,
    valDf: DataFrame
  ): Seq[(String, Double, Double)] = {
    models.map { case (name, model) =>
      println(s"\nTraining $name...")
      val sparkModel = model.fit(trainDf)
      val preds = sparkModel.transform(valDf)
      val accur = accuracyEvaluator.evaluate(preds)
      val f1Score = f1Evaluator.evaluate(preds)
      println(f"Accuracy: $accur%.4f, \nF1 Score: $f1Score%.4f")
      (name, accur, f1Score)
    }
  }
}

