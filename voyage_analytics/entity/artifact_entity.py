from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    trained_file_path:str 
    test_file_path:str

@dataclass
class DataIngestionArtifactUsers:
    trained_file_path:str 
    test_file_path:str 

@dataclass
class DataIngestionArtifactFlights:
    trained_file_path:str 
    test_file_path:str 

@dataclass
class DataIngestionArtifact:
    trained_file_path:str 
    test_file_path:str

@dataclass
class DataIngestionArtifactHotels:
    trained_file_path:str 
    test_file_path:str


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
    transformed_train_file_path:str
    transformed_test_file_path:str  


@dataclass
class ClassificationMetricArtifact:
    f1_score:float
    precision_score:float
    recall_score:float
    
@dataclass
class RegressionMetricArtifact:
    r2_score:float
    mean_absolute_error:float
    mean_squared_error:float
    root_mean_squared_error:float

@dataclass
class RecumendationMetricArtifact:
    recall:float
    precision:float
    f1-score:float


@dataclass
class ModelTrainerArtifactClassification:
    trained_model_file_path:str 
    metric_artifact:ClassificationMetricArtifact

@dataclass
class ModelTrainerArtifactRegression:
    trained_model_file_path:str 
    metric_artifact:RegressionMetricArtifact
    
@dataclass
class ModelTrainerArtifactRecumendation:
    trained_model_file_path:str 
    metric_artifact:RecumendationMetricArtifact

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



