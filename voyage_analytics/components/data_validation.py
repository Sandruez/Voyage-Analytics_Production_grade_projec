import json
import sys

import pandas as pd
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection

from pandas import DataFrame

from voyage_analytics.exception import VoyageAnalyticsException
from voyage_analytics.logger import logging
from voyage_analytics.utils.main_utils import read_yaml_file, write_yaml_file
from voyage_analytics.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from voyage_analytics.entity.config_entity import DataValidationConfig
from voyage_analytics.entity.config_entity import SchemaConfig 



class DataValidation:
    def __init__(self, data_ingestion_artifact:DataIngestionArtifact, data_validation_config: DataValidationConfig):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_validation_config: configuration for data validation
        """
        try:
            self.flight_data_ingestion_artifact = data_ingestion_artifact.flight_data_ingestion_artifact
            self.hotel_data_ingestion_artifact = data_ingestion_artifact.hotel_data_ingestion_artifact
            self.user_data_ingestion_artifact = data_ingestion_artifact.user_data_ingestion_artifact
            
            self.data_validation_config = data_validation_config
           
            
            
        except Exception as e:
            raise VoyageAnalyticsException(e,sys)
        
    def get_schema(self, task: str):
        """
        Method Name :   get_schema
        Description :   This method reads the schema file and returns the schema content
        
        Output      :   Returns schema content
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            schema_file_path = getattr(SchemaConfig(), task)
            _schema_config = read_yaml_file(file_path=schema_file_path)
            return _schema_config
        except Exception as e:
            raise VoyageAnalyticsException(e, sys) from e

    def validate_number_of_columns(self, dataframe: DataFrame, task: str) -> bool:
        """
        Method Name :   validate_number_of_columns
        Description :   This method validates the number of columns
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            _schema_config = self.get_schema(task=task)
            status = len(dataframe.columns) == len(_schema_config["columns"])
            logging.info(f"Is required column present: [{status}]")
            return status
        except Exception as e:
            raise VoyageAnalyticsException(e, sys)

    def is_column_exist(self, df: DataFrame, task: str) -> bool:
        """
        Method Name :   is_column_exist
        Description :   This method validates the existence of a numerical and categorical columns
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            _schema_config = self.get_schema(task=task)
            
            dataframe_columns = df.columns
            missing_numerical_columns = []
            missing_categorical_columns = []
            for column in _schema_config["numerical_columns"]:
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)

            if len(missing_numerical_columns)>0:
                logging.info(f"Missing numerical column: {missing_numerical_columns}")


            for column in _schema_config["categorical_columns"]:
                if column not in dataframe_columns:
                    missing_categorical_columns.append(column)

            if len(missing_categorical_columns)>0:
                logging.info(f"Missing categorical column: {missing_categorical_columns}")

            return False if len(missing_categorical_columns)>0 or len(missing_numerical_columns)>0 else True
        except Exception as e:
            raise VoyageAnalyticsException(e, sys) from e

    @staticmethod
    def read_data(file_path) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise VoyageAnalyticsException(e, sys)

    def detect_dataset_drift(self, reference_df: DataFrame, current_df: DataFrame,task: str) -> bool:
        """
        Method Name :   detect_dataset_drift
        Description :   This method validates if drift is detected
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            data_drift_profile = Profile(sections=[DataDriftProfileSection()])

            data_drift_profile.calculate(reference_df, current_df)

            report = data_drift_profile.json()
            json_report = json.loads(report)

            write_yaml_file(file_path=getattr(self.data_validation_config,task), content=json_report)

            n_features = json_report["data_drift"]["data"]["metrics"]["n_features"]
            n_drifted_features = json_report["data_drift"]["data"]["metrics"]["n_drifted_features"]

            logging.info(f"{n_drifted_features}/{n_features} drift detected.")
            drift_status = json_report["data_drift"]["data"]["metrics"]["dataset_drift"]
            return drift_status
        except Exception as e:
            raise VoyageAnalyticsException(e, sys) from e
        
        
        
    def initiate_data_validation(self) -> DataValidationArtifact:
        """_summary_
        Args:
            task (str): _description_
        Raises:
            VoyageAnalyticsException: _description_
        Returns:
            DataValidationArtifact: _description_
        """
        try:
            #Users - class_
            #Flights - reg
            #Hotels - recomend
            # for Users data validation
            user_validation_error_msg = ""
            logging.info("Starting users data validation")

            train_df, test_df = (DataValidation.read_data(file_path=self.user_data_ingestion_artifact.trained_file_path),
                                 DataValidation.read_data(file_path=self.user_data_ingestion_artifact.test_file_path))

            status = self.validate_number_of_columns(dataframe=train_df , task='class_')
            logging.info(f"All required columns present in training user dataframe: {status}")
            if not status:
                user_validation_error_msg += f"Columns are missing in training user dataframe."
            status = self.validate_number_of_columns(dataframe=test_df, task='class_')
            logging.info(f"All required columns present in testing user dataframe: {status}")
            if not status:
                user_validation_error_msg += f"Columns are missing in user test dataframe."

            status = self.is_column_exist(df=train_df , task='class_')
            if not status:
                user_validation_error_msg += f"Columns are missing in training user dataframe."
            status = self.is_column_exist(df=test_df, task='class_')
            if not status:
                user_validation_error_msg += f"columns are missing in test user dataframe."

            user_validation_status = len(user_validation_error_msg) == 0

            if user_validation_status:
                user_drift_status = self.detect_dataset_drift(train_df, test_df, task='class_')
                if user_drift_status:
                    logging.info(f"Drift detected.")
                    user_validation_error_msg = "Drift detected"
                else:
                    user_validation_error_msg = "Drift not detected"
            else:
                logging.info(f"Validation_error: {user_validation_error_msg}")
                
            # for Flights data validation
            flight_validation_error_msg = ""
            logging.info("Starting flights data validation")

            train_df, test_df = (DataValidation.read_data(file_path=self.flight_data_ingestion_artifact.trained_file_path),
                                 DataValidation.read_data(file_path=self.flight_data_ingestion_artifact.test_file_path))

            status = self.validate_number_of_columns(dataframe=train_df , task='reg')
            logging.info(f"All required columns present in training flight dataframe: {status}")
            if not status:
                flight_validation_error_msg += f"Columns are missing in training flight dataframe."
            status = self.validate_number_of_columns(dataframe=test_df, task='reg')
            logging.info(f"All required columns present in testing flight dataframe: {status}")
            if not status:
                flight_validation_error_msg += f"Columns are missing in flight test dataframe."

            status = self.is_column_exist(df=train_df , task='reg')
            if not status:
                flight_validation_error_msg += f"Columns are missing in training flight dataframe."
            status = self.is_column_exist(df=test_df, task='reg')
            if not status:
                flight_validation_error_msg += f"columns are missing in test flight dataframe."

            flight_validation_status = len(flight_validation_error_msg) == 0

            if flight_validation_status:
                flight_drift_status = self.detect_dataset_drift(train_df, test_df, task='reg')
                if flight_drift_status:
                    logging.info(f"Drift detected.")
                    flight_validation_error_msg = "Drift detected"
                else:
                    flight_validation_error_msg = "Drift not detected"
            else:
                logging.info(f"Validation_error: {flight_validation_error_msg}")


            # for hotels data validation
            hotel_validation_error_msg = ""
            logging.info("Starting hotels data validation")

            train_df, test_df = (DataValidation.read_data(file_path=self.hotel_data_ingestion_artifact.trained_file_path),
                                 DataValidation.read_data(file_path=self.hotel_data_ingestion_artifact.test_file_path))

            status = self.validate_number_of_columns(dataframe=train_df , task='recomend')
            logging.info(f"All required columns present in training hotel dataframe: {status}")
            if not status:
                hotel_validation_error_msg += f"Columns are missing in training hotel dataframe."
            status = self.validate_number_of_columns(dataframe=test_df, task='recomend')
            logging.info(f"All required columns present in testing hotel dataframe: {status}")
            if not status:
                hotel_validation_error_msg += f"Columns are missing in hotel test dataframe."

            status = self.is_column_exist(df=train_df , task='recomend')
            if not status:
                hotel_validation_error_msg += f"Columns are missing in training hotel dataframe."
            status = self.is_column_exist(df=test_df, task='recomend')
            if not status:
                hotel_validation_error_msg += f"columns are missing in test hotel dataframe." 
            hotel_validation_status = len(hotel_validation_error_msg) == 0

            if hotel_validation_status:
                hotel_drift_status = self.detect_dataset_drift(train_df, test_df, task='recomend')
                if hotel_drift_status:
                    logging.info(f"Drift detected.")
                    hotel_validation_error_msg = "Drift detected"
                else:
                    hotel_validation_error_msg = "Drift not detected"
            else:
                logging.info(f"Validation_error: {hotel_validation_error_msg}")



            data_validation_artifact = DataValidationArtifact(
                user_data_validation_artifact=DataValidationArtifactUsers(
                    validation_status=user_validation_status,
                    message=validation_error_msg,
                    drift_report_file_path=getattr(self.data_validation_config,'class')
                ),
                flight_data_validation_artifact=DataValidationArtifactFlights(
                    validation_status=flight_validation_status,
                    message=flight_validation_error_msg,
                    drift_report_file_path=getattr(self.data_validation_config,'reg')
                ),
                hotel_data_validation_artifact=DataValidationArtifactHotels(
                    validation_status=hotel_validation_status,
                    message=hotel_validation_error_msg,
                    drift_report_file_path=getattr(self.data_validation_config,'recomend')
                )
            )

            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise USvisaException(e, sys) from e    