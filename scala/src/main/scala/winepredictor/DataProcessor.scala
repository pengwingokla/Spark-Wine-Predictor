package winepredictor

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.functions.{col, count, lit, max}
import org.apache.spark.sql.types.NumericType
import Config.DataConfig

import java.io.File

/**
 * Data processing class.
 * Handles data loading, preprocessing, and feature engineering.
 */
class DataProcessor(spark: SparkSession) {

  /**
   * Load training and validation datasets and combine them.
   * 
   * @param trainCsv Path to training CSV file
   * @param valCsv Path to validation CSV file
   * @return Combined dataset
   */
  def loadDataset(trainCsv: String, valCsv: String): DataFrame = {
    val trainFile = new File(trainCsv)
    val valFile = new File(valCsv)
    
    if (!trainFile.exists() || !valFile.exists()) {
      throw new FileNotFoundException("The dataset files do not exist.")
    }
    
    val trainDf = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .option("sep", DataConfig.CSV_SEPARATOR)
      .csv(trainCsv)
    
    val valDf = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .option("sep", DataConfig.CSV_SEPARATOR)
      .csv(valCsv)
    
    trainDf.union(valDf)
  }

  /**
   * Preprocess the dataset: clean column names, create features, and label.
   * 
   * @param df Input DataFrame
   * @return Preprocessed DataFrame with 'features' and 'label' columns
   */
  def preprocessData(df: DataFrame): DataFrame = {
    // Remove extra quotes from column names
    val cleanedColumns = df.columns.map(_.replace("\"", ""))
    var processedDf = df.toDF(cleanedColumns: _*)
    
    // Features: Exclude the 'quality' column
    val featureCols = processedDf.columns.dropRight(1)
    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")
    
    processedDf = assembler.transform(processedDf)
    
    // Target: Ensure 'quality' column is numeric
    val qualityField = processedDf.schema("quality")
    if (qualityField.dataType.isInstanceOf[NumericType]) {
      // If 'quality' is numeric, rename it to 'label'
      processedDf = processedDf.withColumnRenamed("quality", "label")
    } else {
      // If 'quality' is not numeric, index it
      val indexer = new StringIndexer()
        .setInputCol("quality")
        .setOutputCol("label")
      processedDf = indexer.fit(processedDf).transform(processedDf)
    }
    
    processedDf
  }

  /**
   * Oversample minority classes to balance the dataset.
   * 
   * @param df Input DataFrame with 'label' column
   * @return Oversampled DataFrame
   */
  def oversampleMinorityClasses(df: DataFrame): DataFrame = {
    // Check class distribution
    val totCount = df.count()
    
    println("Original class distribution:")
    val clsDist = df.groupBy("label")
      .agg(count("label").alias("cls_count"))
      .withColumn("percentage", (col("cls_count") / lit(totCount)) * 100)
      .orderBy("label")
    clsDist.show()
    
    // Maximum class count
    val maxCount = clsDist.agg(max("cls_count")).collect()(0)(0).asInstanceOf[Long]
    
    // Oversample minority classes
    var oversampledDf = df
    val clsDistRows = clsDist.collect()
    
    for (row <- clsDistRows) {
      val label = row.getAs[Double]("label")
      val clsCount = row.getAs[Long]("cls_count")
      
      if (clsCount < maxCount) {
        val fraction = (maxCount - clsCount).toDouble / clsCount.toDouble
        val sampledDf = df.filter(col("label") === label)
          .sample(withReplacement = true, fraction = fraction, seed = DataConfig.RANDOM_SEED)
        oversampledDf = oversampledDf.union(sampledDf)
      }
    }
    
    // Verify new class distribution
    println("New class distribution:")
    val newDist = oversampledDf.groupBy("label")
      .agg(count("label").alias("cls_count"))
      .withColumn("percentage", (col("cls_count") / lit(totCount)) * 100)
      .orderBy("label")
    newDist.show()
    
    oversampledDf
  }

  /**
   * Split dataset into training and validation sets.
   * 
   * @param df Input DataFrame
   * @return Tuple of (trainDf, valDf)
   */
  def splitData(df: DataFrame): (DataFrame, DataFrame) = {
    val splits = df.randomSplit(
      Array(DataConfig.TRAIN_RATIO, 1 - DataConfig.TRAIN_RATIO),
      seed = DataConfig.RANDOM_SEED
    )
    (splits(0), splits(1))
  }
}

