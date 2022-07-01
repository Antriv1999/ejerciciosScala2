import org.apache.spark.SparkContext
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.functions.{col, not}

object AATrabajoFinal {
  def main(args: Array[String]): Unit = {
    //Creando el contexto del Servidor
    val sc = new SparkContext("local","AATrabajoFinal", System.getenv("SPARK_HOME"))

    sc.setLogLevel("ERROR")

    val spark = SparkSession
      .builder()
      .master("local")
      .appName("CargaJSON")
      .getOrCreate()

    // Load and parse the data
    val data = spark.read.format("csv").option("header", "true")
      .option("inferSchema", "true").option("delimiter", ",").load("resources/songs_normalize.csv").toDF()

    //data.limit(5).show()

    val stringIndexer = new StringIndexer()
      .setInputCol("artist")
      .setOutputCol("indexedArtist")
      .fit(data)

    val dfindex = stringIndexer.transform(data)

    val stringIndexer2 = new StringIndexer()
      .setInputCol("genre")
      .setOutputCol("indexedGenre")
      .fit(dfindex)

    val dfindex2 = stringIndexer2.transform(dfindex)

    val df = dfindex2.withColumn("explicit",col("explicit").cast("Integer"))

    val df2 = df.drop("song", "genre", "artist")

    //df2.limit(5).show()

    val featureCols = Array("duration_ms", "explicit", "year", "danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness",
      "instrumentalness", "liveness" , "valence", "tempo", "indexedArtist", "indexedGenre")
    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
    val df3 = assembler.transform(dfindex2)

    val model = new LogisticRegression().setLabelCol("popularity").fit(df3)
    val predictions = model.transform(df3)
    /**
     *  Now we print it out.  Notice that the LR algorithm added a “prediction” column
     *  to our dataframe.   The prediction in almost all cases will be the same as the label.  That is
     * to be expected it there is a strong correlation between these values.  In other words
     * if the chance of getting cancer was not closely related to these variables then LR
     * was the wrong model to use.  The way to check that is to check the accuracy of the model.
     *  You could use the BinaryClassificationEvaluator Spark ML function to do that.
     * Adding that would be a good exercise for you, the reader.
     */
    predictions.select ("features", "popularity", "prediction").show()
    val lp = predictions.select( "popularity", "prediction")
    val counttotal = predictions.count()
    val correct = lp.filter(col("popularity") === col("prediction")).count()
    val wrong = lp.filter(not(col("popularity") === col("prediction"))).count()
    val truep = lp.filter(col("prediction") === 0.0).filter(col("popularity") === col("prediction")).count()
    val falseN = lp.filter(col("prediction") === 0.0).filter(not(col("popularity") === col("prediction"))).count()
    val falseP = lp.filter(col("prediction") === 1.0).filter(not(col("popularity") === col("prediction"))).count()
    val ratioWrong=wrong.toDouble/counttotal.toDouble
    val ratioCorrect=correct.toDouble/counttotal.toDouble

    println(ratioWrong)
    println(ratioCorrect)
    /*val lr = new LinearRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)
      .setLabelCol("popularity")

    // Fit the model
    val lrModel = lr.fit(df3)
    // Print the coefficients and intercept for linear regression
    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

    // Summarize the model over the training set and print out some metrics
    val trainingSummary = lrModel.summary
    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
    trainingSummary.residuals.show()
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    println(s"r2: ${trainingSummary.r2}")*/
  }
}
