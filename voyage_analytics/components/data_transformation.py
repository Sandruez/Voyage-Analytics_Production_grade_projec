import sys

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer,MinMaxScaler,LabelEncoder
from sklearn.compose import ColumnTransformer

from voyage_analytics.constants import TARGET_COLUMN_USERS, TARGET_COLUMN_FLIGHTS, TARGET_COLUMN_HOTELS, CURRENT_YEAR
from voyage_analytics.constants import Users_TARGET_COLUMN, Flights_TARGET_COLUMN, HOTELS_TARGET_COLUMN
from voyage_analytics.entity.config_entity import DataTransformationConfig
from voyage_analytics.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from voyage_analytics.exception import VoyageAnalyticsException
from voyage_analytics.logger import logging
from voyage_analytics.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file, drop_columns,save_dataframe_as_csv
from voyage_analytics.entity.estimator import TargetValueMapping
from voyage_analytics.entity.config_entity import SchemaConfig


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: configuration for data transformation
        """
        try:
            self.user_data_ingestion_artifact = data_ingestion_artifact.user_data_ingestion_artifact
            self.flight_data_ingestion_artifact = data_ingestion_artifact.flight_data_ingestion_artifact
            self.hotel_data_ingestion_artifact = data_ingestion_artifact.hotel_data_ingestion_artifact

            self.data_transformation_config = data_transformation_config
            
            self.user_data_validation_artifact = data_validation_artifact.user_data_validation_artifact
            self.flight_data_validation_artifact = data_validation_artifact.flight_data_validation_artifact
            self.hotel_data_validation_artifact = data_validation_artifact.hotel_data_validation_artifact
            
            self.user_schema_config = read_yaml_file(file_path=getattr(SchemaConfig(), 'class_'))
            self.flight_schema_config = read_yaml_file(file_path=getattr(SchemaConfig(), 'reg'))
            self.hotel_schema_config = read_yaml_file(file_path=getattr(SchemaConfig(), 'recomend'))
        except Exception as e:
            raise VoyageAnalyticsException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise VoyageAnalyticsException(e, sys)



    def get_flight_data_transformer_object(self) -> ColumnTransformer:
        """
        Method Name :   get_flight_data_transformer_object
        Description :   This method creates and returns a data transformer object for the data
        
        Output      :   data transformer object is created and returned 
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Got numerical cols from schema config")

            numeric_transformer = StandardScaler()
            oh_transformer = OneHotEncoder()

            logging.info("Initialized StandardScaler, OneHotEncoder, OrdinalEncoder")

            oh_columns = self.flight_schema_config['oh_columns']
            num_features = self.flight_schema_config['scaler_columns']

            logging.info("Initialize ColumnTransformer")

            preprocessor = ColumnTransformer(
                [
                    ("StandardScaler", numeric_transformer, num_features),
                    ("OneHotEncoder", oh_transformer, oh_columns),
                ]
            )

            logging.info("Created preprocessor object from ColumnTransformer")

            logging.info(
                "Exited get_flight_data_transformer_object method of DataTransformation class"
            )
            return preprocessor

        except Exception as e:
            raise VoyageAnalyticsException(e, sys) from e

    def get_hotel_data_transformer_object(self) -> ColumnTransformer:
        """
        Method Name :   get_hotel_data_transformer_object
        Description :   This method creates and returns a data transformer object for the data
        
        Output      :   data transformer object is created and returned 
        On Failure  :   Write an exception log and then raise an exception
        """

        try:
            logging.info("Got numerical cols from schema config")

            min_max_scaler = MinMaxScaler()
            label_encoder = LabelEncoder()

            logging.info("Initialized MinMaxScaler, LabelEncoder")

            min_max_scale_columns = self.hotel_schema_config['min_max_scale_columns']
            label_encode_columns = self.hotel_schema_config['label_encode_columns']
           
            logging.info("Initialize ColumnTransformer")


            preprocessor = ColumnTransformer(
                [
                    ("LabelEncoder", label_encoder, label_encode_columns),
                    ("MinMaxScaler", min_max_scaler, min_max_scale_columns),
                ]
            )

            logging.info("Created preprocessor object from ColumnTransformer")

            logging.info(
                "Exited get_hotel_data_transformer_object method of DataTransformation class"
            )
            return preprocessor

        except Exception as e:
            raise VoyageAnalyticsException(e, sys) from e
        
# def initiate_data_transformation(self, ) -> DataTransformationArtifact:
    
    def classify_data_transformation(self, ) -> DataTransformationArtifact:
        """
        Method Name :   initiate_data_transformation
        Description :   This method initiates the data transformation component for the pipeline 
        
        Output      :   data transformer steps are performed and preprocessor object is created  
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            if self.flight_data_validation_artifact.validation_status:
                logging.info("Starting user data transformation")
                logging.info("NO any preprocessor object")

                train_df = DataTransformation.read_data(file_path=self.user_data_ingestion_artifact.trained_file_path)
                test_df = DataTransformation.read_data(file_path=self.user_data_ingestion_artifact.test_file_path)

                train_df['name']=train_df['name'].astype('str').str.strip()
                test_df['name']=test_df['name'].astype('str').str.strip()
                logging.info("Converted name column to string type and stripped whitespace")

                input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN_USERS], axis=1)
                target_feature_train_df = train_df[TARGET_COLUMN_USERS]

                logging.info("Got train features and test features of Training dataset")

                drop_cols = self.user_schema_config['drop_columns']

                logging.info("drop the columns in drop_cols of Training dataset")

                input_feature_train_df = drop_columns(df=input_feature_train_df, cols = drop_cols)
                logging.info("Dropped specified columns from training feature set")


                input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN_USERS], axis=1)
                target_feature_test_df = test_df[TARGET_COLUMN_USERS]
                logging.info("Got train features and test features of Testing dataset")

                input_feature_test_df = drop_columns(df=input_feature_test_df, cols = drop_cols)

                logging.info("drop the columns in drop_cols of Test dataset")
                

                logging.info("Got train features and test features of Testing dataset")

                logging.info(
                    "Applying preprocessing object on training dataframe and testing dataframe"
                )
                
                
                input_feature_train_final = input_feature_train_df
                target_feature_train_final = target_feature_train_df
                input_feature_test_final = input_feature_test_df
                target_feature_test_final = target_feature_test_df

                logging.info("Created train and test dataframes")

                train_arr = np.c_[
                    input_feature_train_final, np.array(target_feature_train_final)
                ]

                test_arr = np.c_[
                    input_feature_test_final, np.array(target_feature_test_final)
                ]

                save_object(self.data_transformation_config.transformed_reg_object_file_path, preprocessor)
                save_numpy_array_data(self.data_transformation_config.transformed_reg_train_file_path, array=train_arr)
                save_numpy_array_data(self.data_transformation_config.transformed_reg_test_file_path, array=test_arr)

                logging.info("Saved the preprocessor object")

                logging.info(
                    "Exited initiate_data_transformation method of Data_Transformation class"
                )

                data_transformation_artifact_flights = DataTransformationArtifactFlights(
                    transformed_object_file_path=self.data_transformation_config.transformed_reg_object_file_path,
                    transformed_train_file_path=self.data_transformation_config.transformed_reg_train_file_path,
                    transformed_test_file_path=self.data_transformation_config.transformed_reg_test_file_path
                )
                return data_transformation_artifact_flights
            else:
                raise Exception(self.data_validation_artifact.message)

        except Exception as e:
            raise VoyageAnalyticsException(e, sys) from e
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def reg_data_transformation(self, ) -> DataTransformationArtifact:
        """
        Method Name :   initiate_data_transformation
        Description :   This method initiates the data transformation component for the pipeline 
        
        Output      :   data transformer steps are performed and preprocessor object is created  
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            if self.flight_data_validation_artifact.validation_status:
                logging.info("Starting flight data transformation")
                preprocessor = self.get_flight_data_transformer_object()
                logging.info("Got the preprocessor object")

                train_df = DataTransformation.read_data(file_path=self.flight_data_ingestion_artifact.trained_file_path)
                test_df = DataTransformation.read_data(file_path=self.flight_data_ingestion_artifact.test_file_path)

                train_df["date"]=pd.to_datetime(train_df["date"])
                train_df["day"]=train_df["date"].dt.day_name()
                
                test_df["date"]=pd.to_datetime(test_df["date"])
                test_df["day"]=test_df["date"].dt.day_name()
                
                logging.info("Converted date column to datetime format and extracted day of week")
                
                input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN_FLIGHTS], axis=1)
                target_feature_train_df = train_df[TARGET_COLUMN_FLIGHTS]

                logging.info("Got train features and test features of Training dataset")

                drop_cols = self.flight_schema_config['drop_columns']

                logging.info("drop the columns in drop_cols of Training dataset")

                input_feature_train_df = drop_columns(df=input_feature_train_df, cols = drop_cols)
                logging.info("Dropped specified columns from training feature set")


                input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN_FLIGHTS], axis=1)
                target_feature_test_df = test_df[TARGET_COLUMN_FLIGHTS]
                logging.info("Got train features and test features of Testing dataset")

                input_feature_test_df = drop_columns(df=input_feature_test_df, cols = drop_cols)

                logging.info("drop the columns in drop_cols of Test dataset")
                

                logging.info("Got train features and test features of Testing dataset")

                logging.info(
                    "Applying preprocessing object on training dataframe and testing dataframe"
                )

                input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)

                logging.info(
                    "Used the preprocessor object to fit transform the train features"
                )

                input_feature_test_arr = preprocessor.transform(input_feature_test_df)

                input_feature_train_final = input_feature_train_arr
                target_feature_train_final = target_feature_train_df.values
                input_feature_test_final = input_feature_test_arr
                target_feature_test_final = target_feature_test_df.values

                logging.info("Used the preprocessor object to transform the test features")
                logging.info("Created train array and test array")

                train_arr = np.c_[
                    input_feature_train_final, np.array(target_feature_train_final)
                ]

                test_arr = np.c_[
                    input_feature_test_final, np.array(target_feature_test_final)
                ]

                save_object(self.data_transformation_config.transformed_reg_object_file_path, preprocessor)
                save_numpy_array_data(self.data_transformation_config.transformed_reg_train_file_path, array=train_arr)
                save_numpy_array_data(self.data_transformation_config.transformed_reg_test_file_path, array=test_arr)

                logging.info("Saved the preprocessor object")

                logging.info(
                    "Exited initiate_data_transformation method of Data_Transformation class"
                )

                data_transformation_artifact_flights = DataTransformationArtifactFlights(
                    transformed_object_file_path=self.data_transformation_config.transformed_reg_object_file_path,
                    transformed_train_file_path=self.data_transformation_config.transformed_reg_train_file_path,
                    transformed_test_file_path=self.data_transformation_config.transformed_reg_test_file_path
                )
                return data_transformation_artifact_flights
            else:
                raise Exception(self.data_validation_artifact.message)

        except Exception as e:
            raise VoyageAnalyticsException(e, sys) from e