import os
from pymongo import MongoClient
from dotenv import load_dotenv
import requests
from pyspark.sql import SparkSession
from pyspark.sql.functions import window, col, lag, monotonically_increasing_id
from pyspark.sql.types import StructType, StructField, FloatType, IntegerType, TimestampType, StringType
import yaml


load_dotenv()

def initiate_spark():   
    """
    Initialize SparkSession.

    """
    try:
        spark = SparkSession.builder \
            .appName("LocalSparkSession") \
            .master("local[*]") \
            .config("spark.network.timeout", "600s") \
            .config("spark.sql.debug.maxToStringFields", 100) \
            .getOrCreate()
        return spark
    except Exception as e:
        print("error in initializing sparksession :",e)

def terminate_spark(spark):
    try:
        spark.stop()
    except Exception as e:
        print("error in terminatating sparksession :",e)



def timeseries_train_test_split(df, training_ratio, test_size, gap_size):
    """
    This function splits a PySpark DataFrame containing time series data 
    into training and testing sets while maintaining temporal order.

    Args:
        df: PySpark DataFrame containing your time series data
        training_ratio: Proportion of data allocated to training (0.0 to 1.0)
        test_size: Number of time steps for the testing set
        gap_size: Number of time steps between the training and testing sets

    Returns:
        A tuple containing two PySpark DataFrames (training_data, testing_data)
    """

    # Calculate training data size based on ratio
    training_size = int(df.count() * training_ratio)
    print("training_size:", training_size)
    # Add a monotonically increasing ID for ordering
    df = df.withColumn("row_id", monotonically_increasing_id())
    df.show(5)
    # Window function to identify rows within the testing window
    window_spec = window.orderBy("row_id").partitionBy().rowsBetween(-test_size - gap_size, -1)
    df_with_test_flag = df.withColumn("is_test", col("row_id").isin(window_spec))

    # Filter training and testing data based on the flag
    training_data = df.where(~col("is_test"))
    testing_data = df.where(col("is_test"))

    # Ensure training data covers training_size rows
    training_data = training_data.limit(training_size)

    return training_data, testing_data



def load_schema_from_yaml(file_path):
    """
    Load schema from YAML file.
    
    Args:
    - file_path: Path to the YAML file containing the schema.
    
    Returns:
    - StructType: The schema loaded from the YAML file.
    """
    with open(file_path, "r") as f:
        schema_dict = yaml.safe_load(f)

    fields = []
    for field_name, field_type_str in schema_dict.items():
        field_type = eval(field_type_str)()
        fields.append(StructField(field_name, field_type, True))

    return StructType(fields)


def reading_and_validate_data(spark, data, schema_file_path):
    """
    Validates the given data against the provided schema.
    
    Args:
    - spark: The SparkSession object.
    - data: The data to be validated.
    - schema_file_path: Path to the YAML file containing the schema.
    
    Returns:
    - DataFrame: The validated DataFrame if successful, else None.
    """
    try:
        # Load schema from YAML file
        schema = load_schema_from_yaml(schema_file_path)

        # Create DataFrame using provided schema
        df = spark.createDataFrame(data=data, schema=schema)
        
        if df is not None:
            return df
        else:
            print("Empty DataFrame to validate.")
            return None
    except Exception as e:
        print("Validation Error:", e)
        return None

def train_test_split(df):
    try:
        total_rows = df.count()
        test_rows = int(0.2 * total_rows)
        train_rows = max(0, total_rows - test_rows)
        test = df.orderBy(col("creation_time").desc()).limit(test_rows)
        train = df.orderBy(col("creation_time").asc()).limit(train_rows)
        return [(train,test)]
    except Exception as e:
        print("error in spliting data:",e)

"""
##############                    mongodb functions              ###########################
"""
def get_connection():
    try:
        db = os.getenv("db")
        host = os.getenv("host")
        port = os.getenv("port")
        mongo_url = f"mongodb://{host}:{port}"
        client = MongoClient(mongo_url)
        db1 = client[db]
        return db1

    except Exception as e:
        print("Error in Mongo_conection", e)
        
def get_data_from_api():
    try:
        url = 'https://api.example.com/data'
        with requests.post(url) as response:
            response.raise_for_status()  # Raise exception for HTTP errors
            # Parse the JSON response
            data = response.json()

            print(data)
    
    except requests.exceptions.RequestException as e:
        # Catch any requests-related exceptions
        print("Error while fetching data:", e)
    
    except ValueError as e:
        # Catch JSON decoding errors
        print("Error decoding JSON:", e)


def get_circle_data():
    try:
        dataList = []
        conn = get_connection()
        collection_name = os.getenv("circle")
        circle = conn[collection_name]
        data = circle.find({"utility": "2"},{"_id":0,"id":1})
        dataList.extend(data)
        # print(dataList)
        return dataList
    except Exception as e:
        print("Error fetching circle data from MongoDB:", e)
        
def get_sensor_data(circle):
    try:
        dataList = []
        conn = get_connection()
        collection_name = os.getenv("sensor")
        sensor = conn[collection_name]
        data = sensor.find({"circle_id": circle, "type": "AC_METER", "admin_status": {"$in": ['N', 'S', 'U']}, "utility": "2"},
                           {"name": 1, "_id": 0, "id": 1, "meter_ct_mf": 1, "UOM": 1, "meter_MWh_mf": 1, "site_id": 1, "asset_id": 1}).limit(1000)
        for item in data:
            dataList.append(item)
        return dataList
    except Exception as e:
        print("Error fetching data from MongoDB:", e)

def get_load_profile_data(circle_id):
    try:
        dataList = []
        conn = get_connection()
        collection_name = os.getenv("loadProfileData")
        loadProfile = conn[collection_name]
        sensor_lst = get_sensor_data(circle_id)
        for sensorId in sensor_lst:
            fromId = sensorId + "-2024-01-01 00:00:00"
            toId = sensorId + "-2024-03-31 23:59:59"
            data = loadProfile.find({"_id": {"$gte": fromId, "$lte": toId}})
        # data  = loadProfile.find({"circle_id":circle_id},{"_id":0,"data":1})
        for doc in data:
            dataList.append(doc['data'])
        print(len(dataList))
        return dataList
    except Exception as e:
        print("Error fetching data from MongoDB:", e)

def get_process_sensor(circle_id):
    try:
        dataList = []
        conn = get_connection()
        collection_name = os.getenv("transformed_data")
        loadProfile = conn[collection_name]
        # fromId = sensorId + "-2024-01-01 00:00:00"
        # toId = sensorId + "-2024-03-31 23:59:59"
        # data = loadProfile.find({"_id": {"$gte": fromId, "$lte": toId}})
        data  = loadProfile.find({"circle_id":circle_id},{"_id":0,"data":1})
        for doc in data:
            dataList.append(doc['data'])
        print(len(dataList))
        return dataList
    except Exception as e:
        print("Error fetching data from MongoDB:", e)

# zero_counter = 0
# for i in data:
#     # print(id_counter,end=" ")
#     s_data = list(collection1.find({"_id": {"$gt": f"{i['id']}-2024-01-01 00:00:00",
#           "$lt": f"{i['id']}-2024-12-31 23:59:59"}}))
#     # s_data = list(collection1.find({"sensor_id": s_id["id"]}, {'creation_time', 'sensor_id', 'opening_KWh', 'closing_KWh'}))
#     if len(s_data)!= 0:
#         df = pd.DataFrame(s_data)
#         circle_df_lst.append(df) 
#         # break
#     else:
#         zero_counter+=1   
# circle_df = pd.concat(circle_df_lst) 