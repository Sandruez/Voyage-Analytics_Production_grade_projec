import os
from datetime import date

DATABASE_NAME = "user_flights_hotels_DB"

COLLECTION_USERS_NAME = "users"
COLLECTION_flights_NAME = "flights"
COLLECTION_HOTELS_NAME = "hotels"

MONGODB_URL_KEY = "MONGODB_URL"

PIPELINE_NAME: str = "VoyageAnalyticsPipeline"
ARTIFACT_DIR: str = "artifact"


TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

FILE_NAME_USERS: str = "user.csv"
FILE_NAME_flights: str = "flights.csv"
FILE_NAME_HOTELS: str = "hotels.csv"
MODEL_FILE_NAME = "model.pkl"

SCHEMA_USERS_FILE_PATH = os.path.join("config", "users_schema.yaml")
SCHEMA_Flights_FILE_PATH = os.path.join("config", "flights_schema.yaml")
SCHEMA_HOTELS_FILE_PATH = os.path.join("config", "hotels_schema.yaml")



TARGET_COLUMN_USERS = "gender"
TARGET_COLUMN_flights = "price"
TARGET_COLUMN_HOTELS = ""
CURRENT_YEAR = date.today().year

PREPROCSSING_OBJECT_REGRESSION_FILE_NAME = "reg_preprocessing.pkl"
PREPROCSSING_OBJECT_CLASSIFICATION_FILE_NAME = "class_preprocessing.pkl"
PREPROCSSING_OBJECT_RECUMENDATION_FILE_NAME = "recumend_preprocessing.pkl"


USERS_SCHEMA_FILE_PATH = os.path.join("config", "users_schema.yaml")
FLIGHTS_SCHEMA_FILE_PATH = os.path.join("config", "flights_schema.yaml")
HOTELS_SCHEMA_FILE_PATH = os.path.join("config", "hotels_schema.yaml")

USERS_FILE_NAME: str = "users.csv"
FLIGHTS_FILE_NAME: str = "flights.csv"
HOTELS_FILE_NAME: str = "hotels.csv"

TASK1: str = "regression"
TASK2: str = "classification"
TASK3: str = "recumendation"

AWS_ACCESS_KEY_ID_ENV_KEY = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ENV_KEY = "AWS_SECRET_ACCESS_KEY"
REGION_NAME = "us-east-1"



"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_USERS_COLLECTION_NAME: str = "Users"
DATA_INGESTION_flights_COLLECTION_NAME: str = "flights"
DATA_INGESTION_HOTELS_COLLECTION_NAME: str = "hotels"

DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"


DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2


"""
Data Validation realted constant start with DATA_VALIDATION VAR NAME
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"

DATA_VALIDATION_REG_DRIFT_REPORT_FILE_NAME: str = "reg_report.yaml"
DATA_VALIDATION_CLASS_DRIFT_REPORT_FILE_NAME: str = "class_report.yaml"
DATA_VALIDATION_RECUMEND_DRIFT_REPORT_FILE_NAME: str = "recumend_report.yaml"



"""
Data Transformation ralated constant start with DATA_TRANSFORMATION VAR NAME
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"


"""
MODEL TRAINER related constant start with MODEL_TRAINER var name
"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join("config", "model.yaml")


"""
MODEL EVALUATION related constant 
"""
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
MODEL_BUCKET_NAME = "usvisa-model2024"
MODEL_PUSHER_S3_KEY = "model-registry"


APP_HOST = "0.0.0.0"
APP_PORT = 8080

