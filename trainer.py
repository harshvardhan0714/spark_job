from pyspark.sql import Window

import os
from dotenv import load_dotenv
load_dotenv()
import requests
from spark_common import initiate_spark,terminate_spark, get_circle_data , get_process_sensor, get_data_from_api,reading_and_validate_data,train_test_split
from pyspark.sql.types import StructType, StructField, StringType, FloatType, TimestampType, IntegerType
from pyspark.sql.functions import *

import pandas as pd
import sys
from queue import Queue
from threading import Thread

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# from pyspark.ml.regression import XGBRegressor
from xgboost import XGBRegressor

from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer
import joblib
from pyspark.sql.functions import pandas_udf, PandasUDFType
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV


circle_list = get_circle_data()
# data = get_process_sensor(circle_list[0]['id'])
data= []
for id in range(len(circle_list)):
    data_complete  = get_process_sensor(circle_list[id]['id'])
    data.extend(data_complete )


def model_training(spark):
    try:     

        result_schema = StructType([
            StructField("temperature_2m", FloatType(), True),
            StructField("relative_humidity_2m", IntegerType(), True),
            StructField("apparent_temperature", FloatType(), True),
            StructField("precipitation", FloatType(), True),
            StructField("wind_speed_10m", FloatType(), True),
            StructField("wind_speed_100m", FloatType(), True),
            StructField("is_holiday", IntegerType(), True),
            StructField("lag1", FloatType(), True),
            StructField("lag2", FloatType(), True),
            StructField("lag3", FloatType(), True),
            StructField("lag4", FloatType(), True),
            StructField("lag5", FloatType(), True),
            StructField("day", IntegerType(), True),
            StructField("hour", IntegerType(), True),
            StructField("month", IntegerType(), True),
            StructField("dayofweek", IntegerType(), True),
            StructField("quarter", IntegerType(), True),
            StructField("dayofyear", IntegerType(), True),
            StructField("encoded_sensor_id", FloatType(), True),
            StructField("weekofyear", IntegerType(), True),
            StructField("year", IntegerType(), True),
        ])
        schema_file_path = "schema.yaml"
        # Validate the data against the schema
        validated_df = reading_and_validate_data(spark, data, schema_file_path)
        # df = validated_df.select("creation_time","sensor_id","consumed_unit")
        df = validated_df
        print(spark)
        # Assuming you have 'repartitioned_df' containing sensor data
        indexer = StringIndexer(inputCol="sensor_id", outputCol="encoded_sensor_id")
        encoded_df = indexer.fit(df).transform(df)
        numbers_to_partitions = encoded_df.select("encoded_sensor_id").distinct().count()
        print(encoded_df.select("sensor_id").distinct().count())
        print(numbers_to_partitions)

        df = encoded_df.drop("sensor_id")
        num = df.rdd.getNumPartitions()
        print("number of partion before:",num)
        repartitoned_df = df.repartition(numbers_to_partitions,['sensor_id'])
        num1 = repartitoned_df.rdd.getNumPartitions()
        print("number of partion after:",num1)

        def split_train_test(key,pdf):
            total_rows = len(pdf)
            train_rows = int(0.8 * total_rows)
            pdf = pdf.sort_values(by="creation_time")
            train_df = pdf.iloc[:train_rows]
            FEATURES = ['is_holiday','temperature_2m', 'relative_humidity_2m', 'apparent_temperature','precipitation', 'wind_speed_10m','wind_speed_100m', 'lag1', 'lag2','lag3', 'lag4', 'lag5', 'day', 'hour', 'month', 'dayofweek', 'quarter','dayofyear', "encoded_sensor_id",'weekofyear', 'year']
            TARGET = ['consumed_unit']
            
            X_train = train_df[FEATURES]
            y_train = train_df[TARGET]

            # from xgboost import XGBRegressor
            xgb_model = XGBRegressor()
            xgb_model.fit(X_train, y_train)
            param_grid = {
            'n_estimators': [50, 100, 150, 200, 500],
            'max_depth': [5, 10, 15, 20, 30],
            'learning_rate': [0.01, 0.1, 0.3, 0.5, 0.7],
            'reg_alpha':  [0.001, 0.01, 0.1, 0.5, 1.0]
            }

            random_search = RandomizedSearchCV(xgb_model,
                                                param_distributions=param_grid,
                                                n_iter=10,
                                                scoring='neg_mean_squared_error',
                                                cv=5,
                                                # verbose=1,
                                                n_jobs=-1,
                                                random_state=45)

            # Fit the RandomizedSearchCV to the data
            random_search.fit(X_train, y_train)

            # Get the best parameters
            best_params = random_search.best_params_
            # print(f"Best Parameters for sensor {i}: {best_params}")

            # Train the model with the best parameters
            best_xgb_model = XGBRegressor(n_estimators=best_params['n_estimators'],
                                        max_depth=best_params['max_depth'],
                                        learning_rate=best_params['learning_rate'],
                                        reg_alpha=best_params['reg_alpha'],
                                        reg_lambda=0.01,
                                        )
            best_xgb_model.fit(X_train, y_train)
            model_filename = f"xgb_model_sensor_{key}.joblib"
            joblib.dump(xgb_model, model_filename)
            return X_train
        # result = (store_part.groupBy("sensor_id").apply(train_test_split))
        result = repartitoned_df.groupBy("encoded_sensor_id").applyInPandas(split_train_test,schema=result_schema)
        result.show()
    except Exception as e:
        print("error in traininig:",e)

spark = initiate_spark()

if spark:
    try:
        # training data
        model_training(spark)
        pass

    finally:
        # Terminate SparkSession
        # terminate_spark(spark)
        pass