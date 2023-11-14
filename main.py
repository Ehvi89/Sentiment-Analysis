from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF, StringIndexer
from pyspark.sql import SparkSessions

### Create a spark session
spark = SparkSession.builder.appName("SetimentAnalysis").getOrCreate()
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", 5 * 1024 * 1024)  # 5 MiB


### Import the dataset
dataset = spark.read.csv("Twitter_Data.csv")
dataset = dataset.withColumnRenamed("_c0", "tweet").withColumnRenamed("_c1", "label").na.drop()

# another dataset
# data = spark.read.csv("french_tweets.csv", inferSchema=True)
# data = data.withColumnRenamed("_c0", "label").withColumnRenamed("_c1", "tweet").na.drop()

# Show the dataset
# data.show(10)

### Features preparations

# Tokenize the tweets
tokenize = Tokenizer(inputCol="tweet", outputCol="words")
tokenized = tokenize.transform(dataset)

# Remove the unimportant words
stopRemove = StopWordsRemover(inputCol="words", outputCol="filtered")
stopRemoved = stopRemove.transform(tokenized)

# Vectorize words
vectorize = CountVectorizer(inputCol="filtered", outputCol="countVec")
counted_vec = vectorize.fit(stopRemoved).transform(stopRemoved)

idf = IDF(inputCol="countVec", outputCol="features")
rescale = idf.fit(counted_vec).transform(counted_vec)

# Index the label column
indexer = StringIndexer(inputCol="label", outputCol="Index")
indexed = indexer.fit(rescale).transform(rescale)
# indexed.show()

### Prediction & evaluation

# Train and test split
(training, testing) = indexed.randomSplit([0.8, 0.2], seed=42)

# Fit the model
LoReg = LogisticRegression(featuresCol="features", labelCol="Index")
LoRegModel = LoReg.fit(training)

# test the model
test_pred = LoRegModel.transform(testing)
test_pred.select("features", "label", "Index", "prediction").show(100)
