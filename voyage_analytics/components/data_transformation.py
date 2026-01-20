import sys

import scipy.sparse
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer,MinMaxScaler,LabelEncoder
from sklearn.compose import ColumnTransformer


from voyage_analytics.constants import TARGET_COLUMN_USERS, TARGET_COLUMN_FLIGHTS, TARGET_COLUMN_HOTELS, CURRENT_YEAR
from voyage_analytics.entity.config_entity import DataTransformationConfig
from voyage_analytics.entity.artifact_entity import (DataTransformationArtifact, 
                                                     DataIngestionArtifact,
                                                     DataValidationArtifact,
                                                     DataTransformationArtifactUsers,
                                                     DataTransformationArtifactFlights,
                                                     DataTransformationArtifactHotels)
from voyage_analytics.exception import VoyageAnalyticsException
from voyage_analytics.logger import logging
from voyage_analytics.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file, drop_columns,save_df_as_csv
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
            ordinal_encoder = OrdinalEncoder()

            logging.info("Initialized MinMaxScaler, LabelEncoder")

            # 1. Your Lists (based on your request)
            min_max_scale_columns = self.hotel_schema_config['min_max_scale_columns']
            label_encode_columns = self.hotel_schema_config['label_encode_columns']
           
          
            # 2. Dynamic Sorting Logic
            # Find 'place' because it is in BOTH lists
            overlap_cols = list(set(label_encode_columns) & set(min_max_scale_columns)) 

            # Find 'price', 'days', 'popularity' because they are ONLY in the scale list
            only_scale_cols = list(set(min_max_scale_columns) - set(overlap_cols))

            # Find columns that are ONLY encoded (none in this case, but good for safety)
            only_encode_cols = list(set(label_encode_columns) - set(overlap_cols))


            # 3. Create the Special Pipeline for 'place'
            # This ensures: String input -> Encoder -> Scaler -> Final Output
            encode_and_scale_pipeline = Pipeline([
                ('encoder', ordinal_encoder), 
                ('scaler', min_max_scaler)
            ])

            logging.info(f"Pipeline created for {overlap_cols} (Encode -> Scale)")
            logging.info(f"Direct Scaling for {only_scale_cols}")

            # 4. Build the Transformer
            preprocessor = ColumnTransformer(
                transformers=[
                    # A. 'place' goes here (Encode -> Scale)
                    ("Encode_Then_Scale", encode_and_scale_pipeline, overlap_cols),
                    
                    # B. 'price', 'days', 'popularity' go here (Scale only)
                    ("Just_MinMaxScale", min_max_scaler, only_scale_cols),
                    
                    # C. Any other columns just for encoding (if any)
                    ("Just_LabelEncode", ordinal_encoder, only_encode_cols),
                ],
                remainder='drop' 
            )

            logging.info("Created preprocessor object from ColumnTransformer")

            logging.info(
                "Exited get_hotel_data_transformer_object method of DataTransformation class"
            )
            return preprocessor

        except Exception as e:
            raise VoyageAnalyticsException(e, sys) from e
        

    def classify_data_transformation(self, ) ->DataTransformationArtifactUsers:
        """
        Method Name :   initiate_data_transformation
        Description :   This method initiates the data transformation component for the pipeline 
        
        Output      :   data transformer steps are performed and preprocessor object is created  
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            if self.user_data_validation_artifact.validation_status:
                logging.info("Starting user data transformation")
                logging.info("NO any preprocessor object")

                                # 1. Read Data
                train_df = DataTransformation.read_data(file_path=self.user_data_ingestion_artifact.trained_file_path)
                test_df = DataTransformation.read_data(file_path=self.user_data_ingestion_artifact.test_file_path)

                # 2. Initial Data Cleaning (String conversion)
                train_df['name'] = train_df['name'].astype('str').str.strip()
                test_df['name'] = test_df['name'].astype('str').str.strip()
                logging.info("Converted name column to string type and stripped whitespace")

                # 3. Drop Unwanted Columns (BEFORE splitting)
                # We do this first because dropping a unique ID column might reveal duplicates
                drop_cols = self.user_schema_config['drop_columns']

                # Note: Applying drop_columns to the whole dataframe
                train_df = drop_columns(df=train_df, cols=drop_cols)
                test_df = drop_columns(df=test_df, cols=drop_cols)
                logging.info(f"Dropped columns {drop_cols} from train and test datasets")

                # 4. Remove Duplicates
                # Now that unique IDs (in drop_cols) are gone, we can safely remove duplicate rows
                shape_before = train_df.shape
                train_final_df = train_df.drop_duplicates().reset_index(drop=True)
                test_final_df = test_df.drop_duplicates().reset_index(drop=True)
                logging.info(f"Removed duplicates. Rows reduced from {shape_before[0]} to {train_df.shape[0]}")

                

                logging.info("Created train and test dataframes")

                save_df_as_csv(self.data_transformation_config.transformed_class_train_file_path, train_final_df)
                save_df_as_csv(self.data_transformation_config.transformed_class_test_file_path, test_final_df)
                logging.info("Saved the train and test dataframes as csv files")

                logging.info(
                    "Exited classify_data_tranformation method of Data_Transformation class"
                )

                data_transformation_artifact_user = DataTransformationArtifactUsers(
                    transformed_train_file_path=self.data_transformation_config.transformed_class_train_file_path,
                    transformed_test_file_path=self.data_transformation_config.transformed_class_test_file_path,
                    transformed_object_file_path=self.data_transformation_config.transformed_class_object_file_path
                )
                return data_transformation_artifact_user
            else:
                raise Exception(self.user_data_validation_artifact.message)

        except Exception as e:
            raise VoyageAnalyticsException(e, sys) from e
    


    def recommendation_data_transformation(self, ) -> DataTransformationArtifactHotels:
        """
        Method Name :   recommendation_data_transformation
        Description :   recommendation data transformation component for the pipeline 
        
        Output      :   artifact of hotel data transformation is created and returned
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            if self.hotel_data_validation_artifact.validation_status:
                logging.info("Starting hotel data transformation")
                preprocessor = self.get_hotel_data_transformer_object()
                logging.info("Got the preprocessor object")

                train_df = DataTransformation.read_data(file_path=self.hotel_data_ingestion_artifact.trained_file_path)
                test_df = DataTransformation.read_data(file_path=self.hotel_data_ingestion_artifact.test_file_path)


                logging.info("Got train  and test dataframe for hotel data")
                
                whole_df = pd.concat([train_df, test_df], axis=0)
                
                whole_df=drop_columns(df=whole_df, cols=self.hotel_schema_config['drop_columns'])
                logging.info("Dropped dropable column from whole dataframe")
                
                
                
                hotel_data_profiles = whole_df.groupby('name').agg({
                    'place': 'first',
                    'price': 'mean',
                    'days': 'mean',
                    'total':'count'
                }).rename(columns={'total':'popularity'}).reset_index()

                logging.info("Created hotel data profiles based on name")
                
                hotel_features_matrix = preprocessor.fit_transform(hotel_data_profiles.drop(columns=['name'], axis=1))
                
                logging.info("created hotel features matrix using preprocessor object")

                hotel_data_profile_df = hotel_data_profiles.copy()
                hotel_features_matrix_arr = hotel_features_matrix

                save_object(self.data_transformation_config.transformed_recumend_object_file_path, preprocessor)
                save_df_as_csv(self.data_transformation_config.transformed_rec_hotel_data_profile_df_file_path, df=hotel_data_profile_df)
                save_numpy_array_data(self.data_transformation_config.transformed_rec_hotel_features_matrix_arr_file_path, array=hotel_features_matrix_arr)

                logging.info("Saved the preprocessor object and hotel data profile dataframe and hotel features matrix array")

                logging.info(
                    "Exited  recommendation_data_transformation method of Data_Transformation class"
                )

                data_transformation_artifact_hotels = DataTransformationArtifactHotels(
                    transformed_object_file_path=self.data_transformation_config.transformed_recumend_object_file_path,
                    transformed_hotel_data_profile_df_file_path=self.data_transformation_config.transformed_rec_hotel_data_profile_df_file_path,
                    transformed_hotel_features_matrix_arr_file_path=self.data_transformation_config.transformed_rec_hotel_features_matrix_arr_file_path
                )
                return data_transformation_artifact_hotels
            else:
                raise Exception(self.hotel_data_validation_artifact.message)

        except Exception as e:
            raise VoyageAnalyticsException(e, sys) from e
    
    

    def reg_data_transformation(self, ) -> DataTransformationArtifactFlights:
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

               
                 # 1. Read Data
                train_df = DataTransformation.read_data(file_path=self.flight_data_ingestion_artifact.trained_file_path)
                test_df = DataTransformation.read_data(file_path=self.flight_data_ingestion_artifact.test_file_path)

                # 2. Initial Data Cleaning (Date & Day extraction)
                train_df["date"] = pd.to_datetime(train_df["date"])
                train_df["day"] = train_df["date"].dt.day_name()

                test_df["date"] = pd.to_datetime(test_df["date"])
                test_df["day"] = test_df["date"].dt.day_name()
                logging.info("Converted date column to datetime format and extracted day of week")

                # 3. Drop Unwanted Columns (BEFORE splitting)
                # We remove unique IDs or irrelevant columns first so we can identify true duplicates
                drop_cols = self.flight_schema_config['drop_columns']

                # Note: Applying drop_columns to the whole dataframe
                train_df = drop_columns(df=train_df, cols=drop_cols)
                test_df = drop_columns(df=test_df, cols=drop_cols)
                logging.info(f"Dropped columns {drop_cols} from train and test datasets")

                # 4. Remove Duplicates
                # Now we safely remove duplicates, keeping the first occurrence (single entry)
                shape_before = train_df.shape
                train_df = train_df.drop_duplicates()
                test_df = test_df.drop_duplicates()
                logging.info(f"Removed duplicates. Rows reduced from {shape_before[0]} to {train_df.shape[0]}")

                # 5. Split into Input Features and Target Features
                # train split
                input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN_FLIGHTS], axis=1)
                target_feature_train_df = train_df[TARGET_COLUMN_FLIGHTS]

                # test split
                input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN_FLIGHTS], axis=1)
                target_feature_test_df = test_df[TARGET_COLUMN_FLIGHTS]

                logging.info("Got train features and test features of Training and Testing dataset")

                # 6. Preprocessing (Transformations)
                logging.info("Applying preprocessing object on training dataframe and testing dataframe")

                input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
                logging.info("Used the preprocessor object to fit transform the train features")

                input_feature_test_arr = preprocessor.transform(input_feature_test_df)

                # 7. Final Arrays
                input_feature_train_final = input_feature_train_arr
                target_feature_train_final = target_feature_train_df.values
                input_feature_test_final = input_feature_test_arr
                target_feature_test_final = target_feature_test_df.values
               
               # Check if the input features are a sparse matrix (likely yes due to OneHotEncoder)
                
                
                # If sparse, convert to dense array before concatenation
                if scipy.sparse.issparse(input_feature_train_final):
                    input_feature_train_final = input_feature_train_final.toarray()
                    
                if scipy.sparse.issparse(input_feature_test_final):
                    input_feature_test_final = input_feature_test_final.toarray()
               

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
                raise Exception(self.flight_data_validation_artifact.message)

        except Exception as e:
            raise VoyageAnalyticsException(e, sys) from e
    
    
    
    
    def initiate_data_transformation(self, ) -> DataTransformationArtifact:
        """_summary_
        Method Name :   initiate_data_transformation
        Returns:
            DataTransformationArtifact: 
        """
        
        try:
            user_data_transformation_artifact = self.classify_data_transformation()
            flight_data_transformation_artifact = self.reg_data_transformation()
            hotel_data_transformation_artifact = self.recommendation_data_transformation()
            
            data_transformation_artifact = DataTransformationArtifact(
                user_data_transformation_artifact=user_data_transformation_artifact,
                flight_data_transformation_artifact=flight_data_transformation_artifact,
                hotel_data_transformation_artifact=hotel_data_transformation_artifact
            )
            logging.info(
                "Created and returned data transformation artifact"
            )
            return data_transformation_artifact 
        except Exception as e:
            raise VoyageAnalyticsException(e, sys) from e
        
        
        
    