package winepredictor

/**
 * Configuration object for Spark Wine Predictor.
 * Centralizes all configuration settings.
 */
object Config {
  // Dataset paths
  val TRAIN_CSV: String = sys.env.getOrElse("TRAIN_CSV", "TrainingDataset.csv")
  val VAL_CSV: String = sys.env.getOrElse("VAL_CSV", "ValidationDataset.csv")
  val MODEL_PATH: String = sys.env.getOrElse("MODEL_PATH", "best_model")

  // Spark configuration
  object SparkConfig {
    val APP_NAME: String = "WineQualityPrediction"
    val MASTER: String = sys.env.getOrElse("SPARK_MASTER", "spark://172.31.87.32:7077")
    val EXECUTOR_MEMORY: String = sys.env.getOrElse("SPARK_EXECUTOR_MEMORY", "3g")
    val EXECUTOR_CORES: String = sys.env.getOrElse("SPARK_EXECUTOR_CORES", "2")
    val TASK_CPUS: String = sys.env.getOrElse("SPARK_TASK_CPUS", "1")
  }

  // Data processing configuration
  object DataConfig {
    val CSV_SEPARATOR: String = ";"
    val TRAIN_RATIO: Double = 0.8
    val RANDOM_SEED: Long = 42L
  }

  // Model training configuration
  object ModelConfig {
    val NUM_TREES: Array[Int] = Array(20, 50, 75, 100, 150, 200)
    val MAX_DEPTH: Array[Int] = Array(5, 10, 15, 20, 25)
    val MAX_BINS: Array[Int] = Array(16, 32, 48, 64)
    val TRAIN_VALIDATION_SPLIT: Double = 0.8
  }
}

