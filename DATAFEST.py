from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.ml.classification import GBTClassifier  # Import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import os
from tqdm import tqdm
import warnings
from pyspark import SparkConf

# Create a SparkConf with the required configuration
spark_conf = SparkConf().set("spark.sql.autoBroadcastJoinThreshold", "-1") \
                        .set("spark.sql.debug.maxToStringFields", "1000")
# Filter out specific warnings
warnings.filterwarnings("ignore")

def main():
    print('_______________Finding Patterns in the Dataset_______________')
    print(os.environ['JAVA_HOME'])

    # Initialize a Spark session with parallelism settings and the provided SparkConf
    spark = SparkSession.builder.appName("DASE1.0") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executorEnv.JAVA_HOME", os.environ['JAVA_HOME']) \
        .config("spark.default.parallelism", "4") \
        .config(conf=spark_conf).getOrCreate()

    # Load your large CSV file
    csv_file_path = r"C:\Users\hp\Downloads\DATAFEST_HACKATHON\Fraud Detection Dataset.csv"

    # Define the batch size
    batch_size = 10000  # You can adjust this based on available memory

    # Initialize variables for tracking model performance
    overall_accuracy = 0.0
    num_batches = 0

    # Define your model and its parameters (you can adjust these)
    gbt = GBTClassifier(featuresCol="features", labelCol="Fraudulent Flag", maxDepth=10, maxIter=10)  # Use GBTClassifier
    evaluator = BinaryClassificationEvaluator(labelCol="Fraudulent Flag")

    # Columns to be transformed into numerical representations
    categorical_columns = [
        "Payment Method", "Country Code", "Transaction Type", "Device Type", "Browser Type", "Operating System",
        "Merchant Category", "User Occupation", "User Gender", "User Account Status", "Transaction Status",
        "Transaction Time of Day", "User's Device Location", "Transaction Currency", "Transaction Purpose",
        "User's Email Domain", "Transaction Authentication Method"
    ]

    # Remove the "IP Address" column
    drop_columns = ["IP Address", "Transaction Date and Time"]

    # Create a list of stages for the pipeline
    stages = []

    # Iterate over the categorical columns and create indexers and encoders
    for col in categorical_columns:
        indexer = StringIndexer(inputCol=col, outputCol=f"{col}_index")
        encoder = OneHotEncoder(inputCol=f"{col}_index", outputCol=f"{col}_encoded")
        stages += [indexer, encoder]
    print('Numerical Encoding Complete!')

    # Assemble features
    input_features = [f"{col}_encoded" for col in categorical_columns if col not in drop_columns]
    assembler = VectorAssembler(inputCols=input_features, outputCol="features")
    stages.append(assembler)

    # Create a pipeline with all the stages
    pipeline = Pipeline(stages=stages)

    # Iterate over the CSV file in batches
    for i in tqdm(range(0, 6000000, batch_size), desc="Processing Batches", unit=" batches"):
        # Read a batch of data
        batch = spark.read.option("header", "true").option("skipRows", i).csv(csv_file_path, inferSchema=True).limit(
            batch_size)

        # Remove the "IP Address" column
        batch = batch.drop(*drop_columns)
        print(f'Successfully dropped {drop_columns} columns')

        # Cache the batch for reuse
        print("Caching the batch...")
        batch.cache()

        # Fit and transform the data using the pipeline
        transformed_batch = pipeline.fit(batch).transform(batch).select("features", "Fraudulent Flag")

        # Train the model on the batch
        model = gbt.fit(transformed_batch)

        # Optionally, you can evaluate the model's performance on this batch
        predictions = model.transform(transformed_batch)
        batch_accuracy = evaluator.evaluate(predictions)
        overall_accuracy += batch_accuracy
        num_batches += 1

        # Print the accuracy for this batch
        print(f"Batch {num_batches} Accuracy: {batch_accuracy:.2f}%")

        # Unpersist the batch from cache to release memory
        print("Unpersisting the batch...")
        batch.unpersist()

    # Calculate the average accuracy across all batches
    average_accuracy = overall_accuracy / num_batches

    # Print the average accuracy
    print(f"Average Accuracy: {average_accuracy:.2f}%")

    # Stop the Spark session
    spark.stop()

if __name__ == "__main__":
    main()
