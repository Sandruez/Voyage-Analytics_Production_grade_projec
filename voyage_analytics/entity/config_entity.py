import os
from voyage_analytics.constants import *
from dataclasses import dataclass
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    timestamp: str = TIMESTAMP


training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()



@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME)
    
    users_feature_store_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, USERS_FILE_NAME)
    flights_feature_store_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, FLIGHTS_FILE_NAME)
    hotels_feature_store_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, HOTELS_FILE_NAME)

    class_training_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TASK2, TRAIN_FILE_NAME)
    class_testing_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TASK2, TEST_FILE_NAME)
    
    reg_training_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TASK1, TRAIN_FILE_NAME)
    reg_testing_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TASK1, TEST_FILE_NAME)
    
    recumend_training_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TASK3, TRAIN_FILE_NAME)
    recumend_testing_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TASK3, TEST_FILE_NAME)
    
    train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
    
    users_collection_name:str = DATA_INGESTION_USERS_COLLECTION_NAME
    flights_collection_name:str = DATA_INGESTION_flights_COLLECTION_NAME
    hotels_collection_name:str = DATA_INGESTION_HOTELS_COLLECTION_NAME


@dataclass
class SchemaConfig:
    class_: str = USERS_SCHEMA_FILE_PATH
    reg: str = FLIGHTS_SCHEMA_FILE_PATH
    recomend: str = HOTELS_SCHEMA_FILE_PATH


@dataclass
class DataValidationConfig:
    data_validation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR_NAME)
    reg_drift_report_file_path: str = os.path.join(data_validation_dir, DATA_VALIDATION_DRIFT_REPORT_DIR,
                                               DATA_VALIDATION_REG_DRIFT_REPORT_FILE_NAME)
    class_drift_report_file_path: str = os.path.join(data_validation_dir, DATA_VALIDATION_DRIFT_REPORT_DIR,
                                               DATA_VALIDATION_CLASS_DRIFT_REPORT_FILE_NAME)
    recumend_drift_report_file_path: str = os.path.join(data_validation_dir, DATA_VALIDATION_DRIFT_REPORT_DIR,
                                               DATA_VALIDATION_RECUMEND_DRIFT_REPORT_FILE_NAME)
    reg: str =reg_drift_report_file_path
    class_: str =class_drift_report_file_path
    recomend: str =recumend_drift_report_file_path





@dataclass
class DataTransformationConfig:
    data_transformation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_TRANSFORMATION_DIR_NAME)


    transformed_class_train_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,TASK2,
                                                    TRAIN_FILE_NAME)
    transformed_class_test_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,TASK2,
                                                   TEST_FILE_NAME)
    
    transformed_reg_train_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,TASK1,
                                                    TRAIN_FILE_NAME.replace("csv", "npy"))
    transformed_reg_test_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,TASK1,
                                                   TEST_FILE_NAME.replace("csv", "npy"))    
    
    transformed_rec_hotel_data_profile_df_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,TASK3,
                                                    TRAIN_FILE_NAME)
    transformed_rec_hotel_features_matrix_arr_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,TASK3,
                                                   TEST_FILE_NAME.replace("csv", "npy"))
    
    transformed_reg_object_file_path: str = os.path.join(data_transformation_dir,
                                                     DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,TASK1,
                                                     PREPROCSSING_OBJECT_REGRESSION_FILE_NAME)
    transformed_class_object_file_path: str = os.path.join(data_transformation_dir,
                                                     DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,TASK2,
                                                     PREPROCSSING_OBJECT_CLASSIFICATION_FILE_NAME)
    transformed_recumend_object_file_path: str = os.path.join(data_transformation_dir,
                                                     DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,TASK3,
                                                     PREPROCSSING_OBJECT_RECUMENDATION_FILE_NAME)


# artifact\01_20_2026_19_34_29\model_trainer\trained_model\regression
@dataclass
class ModelTrainerConfig:
    model_trainer_dir: str = os.path.join(training_pipeline_config.artifact_dir, MODEL_TRAINER_DIR_NAME)
    trained_reg_model_file_path: str = os.path.join(model_trainer_dir, MODEL_TRAINER_TRAINED_MODEL_DIR,TASK1, MODEL_FILE_NAME)
    trained_class_model_file_path: str = os.path.join(model_trainer_dir, MODEL_TRAINER_TRAINED_MODEL_DIR,TASK2, MODEL_FILE_NAME)
    trained_recumend_model_file_path: str = os.path.join(model_trainer_dir, MODEL_TRAINER_TRAINED_MODEL_DIR,TASK3, MODEL_FILE_NAME)

    expected_accuracy: float = MODEL_TRAINER_EXPECTED_SCORE
    model_config_file_path: str = MODEL_TRAINER_MODEL_CONFIG_FILE_PATH #FOr model hyperparameter tuning yaml file path




@dataclass
class ModelEvaluationConfig:
    changed_threshold_score: float = MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE
    bucket_name: str = MODEL_BUCKET_NAME
    s3_model_key_path: str = MODEL_FILE_NAME



@dataclass
class ModelPusherConfig:
    bucket_name: str = MODEL_BUCKET_NAME
    s3_model_key_path: str = MODEL_FILE_NAME




@dataclass
class USvisaPredictorConfig:
    model_file_path: str = MODEL_FILE_NAME
    model_bucket_name: str = MODEL_BUCKET_NAME





    



