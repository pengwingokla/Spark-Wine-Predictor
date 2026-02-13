package winepredictor

import org.apache.spark.sql.SparkSession
import Config.SparkConfig

/**
 * Spark session initialization object.
 * Handles Spark session creation and configuration.
 */
object SparkSessionFactory {
  
  /**
   * Initialize and return a SparkSession with configured settings.
   * 
   * @return Configured Spark session
   */
  def initializeSpark(): SparkSession = {
    SparkSession.builder()
      .appName(SparkConfig.APP_NAME)
      .master(SparkConfig.MASTER)
      .config("spark.executor.memory", SparkConfig.EXECUTOR_MEMORY)
      .config("spark.executor.cores", SparkConfig.EXECUTOR_CORES)
      .config("spark.task.cpus", SparkConfig.TASK_CPUS)
      .getOrCreate()
  }
}

