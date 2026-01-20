from dataclasses import dataclass


@dataclass
class DataIngestionArtifactUsers:
    trained_file_path:str 
    test_file_path:str 

@dataclass
class DataIngestionArtifactFlights:
    trained_file_path:str 
    test_file_path:str 



@dataclass
class DataIngestionArtifactHotels:
    trained_file_path:str 
    test_file_path:str


@dataclass
class DataIngestionArtifact:
    user_data_ingestion_artifact:DataIngestionArtifactUsers
    flight_data_ingestion_artifact:DataIngestionArtifactFlights
    hotel_data_ingestion_artifact:DataIngestionArtifactHotels



@dataclass
class DataValidationArtifactUsers:
    validation_status:bool
    message: str
    drift_report_file_path: str


@dataclass
class DataValidationArtifactFlights:
    validation_status:bool
    message: str
    drift_report_file_path: str


@dataclass
class DataValidationArtifactHotels:
    validation_status:bool
    message: str
    drift_report_file_path: str

@dataclass
class DataValidationArtifact:
    user_data_validation_artifact:DataValidationArtifactUsers
    flight_data_validation_artifact:DataValidationArtifactFlights
    hotel_data_validation_artifact:DataValidationArtifactHotels


@dataclass
class DataTransformationArtifactUsers:
    transformed_object_file_path:str 
    transformed_train_file_path:str
    transformed_test_file_path:str
    
@dataclass
class DataTransformationArtifactFlights:
    transformed_object_file_path:str 
    transformed_train_file_path:str
    transformed_test_file_path:str
    
@dataclass
class DataTransformationArtifactHotels:
    transformed_object_file_path:str 
    transformed_hotel_data_profile_df_file_path:str
    transformed_hotel_features_matrix_arr_file_path:str

@dataclass
class DataTransformationArtifact:
    user_data_transformation_artifact:DataTransformationArtifactUsers
    flight_data_transformation_artifact:DataTransformationArtifactFlights
    hotel_data_transformation_artifact:DataTransformationArtifactHotels 

@dataclass
class ClassificationMetricArtifact:
    accuracy_score:float
    precision_score:float
    recall_score:float
    f1_score:float
    
@dataclass
class RegressionMetricArtifact:
    r2_score:float
    mse:float
    root_mean_squared_error:float
    mae:float

@dataclass
class RecumendationMetricArtifact:
    recall:float
    precision:float
    f1_score:float


@dataclass
class ModelTrainerArtifactClassification:
    trained_model_file_path:str 
    metric_artifact:ClassificationMetricArtifact

@dataclass
class ModelTrainerArtifactRegression:
    trained_model_file_path:str 
    metric_artifact:RegressionMetricArtifact
    
@dataclass
class ModelTrainerArtifactRecomendation:
    trained_model_file_path:str 
    metric_artifact:RecumendationMetricArtifact


@dataclass
class ModelTrainerArtifact:
    model_trainer_artifact_regression:ModelTrainerArtifactRegression
    model_trainer_artifact_classification:ModelTrainerArtifactClassification
    model_trainer_artifact_recumendation:ModelTrainerArtifactRecumendation



@dataclass
class ModelEvaluationArtifact:
    is_model_accepted:bool
    changed_accuracy:float
    s3_model_path:str 
    trained_model_path:str



@dataclass
class ModelPusherArtifact:
    bucket_name:str
    s3_model_path:str



