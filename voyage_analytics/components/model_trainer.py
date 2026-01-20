import sys
from typing import Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_absolute_error, mean_squared_error, r2_score
from neuro_mf  import ModelFactory

from voyage_analytics.exception import VoyageAnalyticsException
from voyage_analytics.logger import logging
from voyage_analytics.utils.main_utils import load_numpy_array_data, read_yaml_file, load_object, save_object,load_csv_data
from voyage_analytics.entity.config_entity import ModelTrainerConfig
from voyage_analytics.entity.artifact_entity import (DataTransformationArtifact,
                                                     ModelTrainerArtifact,
                                                     ModelTrainerArtifactRegression,
                                                     ModelTrainerArtifactClassification,
                                                     ModelTrainerArtifactRecomendation,
                                                     ClassificationMetricArtifact,
                                                     RegressionMetricArtifact,
                                                     RecumendationMetricArtifact
                                                     )

from voyage_analytics.entity.estimator import RegModel
from voyage_analytics.entity.estimator import ClassificationModel
from voyage_analytics.entity.estimator import RecumendationModel

from voyage_analytics.constants import TARGET_COLUMN_USERS
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: Configuration for data transformation
        """
        self.user_data_transformation_artifact = data_transformation_artifact.user_data_transformation_artifact
        self.flight_data_transformation_artifact = data_transformation_artifact.flight_data_transformation_artifact
        self.hotel_data_transformation_artifact = data_transformation_artifact.hotel_data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    

    def get_reg_model_object_and_report(self, train: np.array, test: np.array) -> Tuple[object, object]:
        """
        Method Name :   getreg_model_object_and_report
        Description :   This function uses neuro_mf to get the best model object and report of the best model
        
        Output      :   Returns metric artifact object and best model object
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Using neuro_mf to get best model object and report")
            model_factory = ModelFactory(model_config_path=self.model_trainer_config.model_config_file_path)
            
            x_train, y_train, x_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]

            best_model_detail = model_factory.get_best_model(
                X=x_train,y=y_train,base_accuracy=self.model_trainer_config.expected_accuracy
            )
            model_obj = best_model_detail.best_model

            y_pred = model_obj.predict(x_test)
            metric_artifact = RegressionMetricArtifact(
                r2_score=r2_score(y_true=y_test, y_pred=y_pred),
                mse=mean_squared_error(y_true=y_test, y_pred=y_pred),
                root_mean_squared_error=np.sqrt(mean_squared_error(y_true=y_test, y_pred=y_pred)),
                mae=mean_absolute_error(y_true=y_test, y_pred=y_pred),
            )
            return best_model_detail, metric_artifact
        
        except Exception as e:
            raise VoyageAnalyticsException(e, sys) from e


    def get_classification_model_object_and_report(self, train_df: DataFrame, test_df: DataFrame) -> Tuple[object, object]:
        """
        Method Name :   get_classification_model_object_and_report
        Description :   This function uses neuro_mf to get the best model object and report of the best model

        Output      :   Returns metric artifact object and best model object
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            

            x_train, y_train, x_test, y_test = train_df.drop(columns=[TARGET_COLUMN_USERS], axis=1), train_df[TARGET_COLUMN_USERS], test_df.drop(columns=[TARGET_COLUMN_USERS], axis=1), test_df[TARGET_COLUMN_USERS]

            model = make_pipeline(
            CountVectorizer(analyzer='char_wb', ngram_range=(2, 4)),
            MultinomialNB()
            )
            x_train.reset_index(drop=True)
            y_train.reset_index(drop=True)
            x_test.reset_index(drop=True)
            y_test.reset_index(drop=True)
            
            logging.info("initiating training the classification model")
            logging.info(f"shape of train_df {x_train.shape},{y_train.shape} and test_df {x_test.shape},{y_test.shape}")
            
            model.fit(x_train.iloc[:, 0], y_train)
            model_obj = model
            logging.info("trained the classification model")

            y_pred = model_obj.predict(x_test.iloc[:,0])
            metric_artifact = ClassificationMetricArtifact(
                accuracy_score = accuracy_score(y_test, y_pred),
                precision_score = precision_score(y_test, y_pred, average='weighted'),
                recall_score = recall_score(y_test, y_pred, average='weighted'),
                f1_score = f1_score(y_test, y_pred, average='weighted')
            )
            return model_obj, metric_artifact
        except Exception as e:
            raise VoyageAnalyticsException(e, sys) from e

    def get_recommendation_model_object_and_report(self, train_arr: np.array) -> Tuple[object, object]:
        """
        Method Name :   get_recommendation_model_object_and_report
        Description :   This function uses neuro_mf to get the best model object and report of the best model
        
        Output      :   Returns metric artifact object and best model object
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("")
            
            profile_matrix_data = train_arr

            model = NearestNeighbors(n_neighbors=7, metric='manhattan',algorithm='brute')
            model.fit(profile_matrix_data)
            logging.info("trained the recommendation model")
            
            
            return model, None
        
        except Exception as e:
            raise VoyageAnalyticsException(e, sys) from e
        

    def initiate_recommendation_model_trainer(self, ) -> ModelTrainerArtifactRecomendation: 
        logging.info("Entered initiate_recommendation_model_trainer method of ModelTrainer class")
        """
        Method Name :   initiate_recommendation_model_trainer
        Description :   This function initiates a model trainer steps
        
        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            profile_matrix_arr = load_numpy_array_data(file_path=self.hotel_data_transformation_artifact.transformed_hotel_features_matrix_arr_file_path)
            hotel_profiles_df=load_csv_data(file_path=self.hotel_data_transformation_artifact.transformed_hotel_data_profile_df_file_path)
            best_model ,_ = self.get_recommendation_model_object_and_report(train_arr=profile_matrix_arr)
            
            logging.info("Created best model file path.")
            
            best_model=RecumendationModel(trained_model_object=best_model,hotel_profiles=hotel_profiles_df, hotel_features_matrix=profile_matrix_arr)
            
            save_object(self.model_trainer_config.trained_recumend_model_file_path , best_model)   
            model_trainer_artifact = ModelTrainerArtifactRecomendation(
                trained_model_file_path=self.model_trainer_config.trained_recumend_model_file_path,
                metric_artifact=None,
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise VoyageAnalyticsException(e, sys) from e   


    def initiate_classification_model_trainer(self, ) -> ModelTrainerArtifactClassification:
        logging.info("Entered initiate_classification_model_trainer method of ModelTrainer class")
        """
        Method Name :   initiate_classification_model_trainer
        Description :   This function initiates a model trainer steps
        
        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            train_df = load_csv_data(file_path=self.user_data_transformation_artifact.transformed_train_file_path)
            test_df = load_csv_data(file_path=self.user_data_transformation_artifact.transformed_test_file_path) 
            
            best_model ,metric_artifact = self.get_classification_model_object_and_report(train_df=train_df, test_df=test_df)
            preprocessing_obj = None
            
            # if best_model.best_score < self.model_trainer_config.expected_accuracy:
            #     logging.info("No best model found with score more than base score")
            #     raise Exception("No best model found with score more than base score")
            class_model = ClassificationModel(trained_model_object=best_model)
            
            logging.info("Created class model object with preprocessor and model")
            logging.info("Created best model file path.")
            save_object(self.model_trainer_config.trained_class_model_file_path , class_model)   
            model_trainer_artifact = ModelTrainerArtifactClassification(
                trained_model_file_path=self.model_trainer_config.trained_class_model_file_path,
                metric_artifact=metric_artifact,
            )
            logging.info(f"Classification evalution metrices:{metric_artifact}")
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise VoyageAnalyticsException(e, sys) from e   
    

    def initiate_reg_model_trainer(self, ) -> ModelTrainerArtifactRegression:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates a model trainer steps
        
        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            train_arr = load_numpy_array_data(file_path=self.flight_data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.flight_data_transformation_artifact.transformed_test_file_path)
            
            best_model_detail ,metric_artifact = self.get_reg_model_object_and_report(train=train_arr, test=test_arr)
            
            preprocessing_obj = load_object(file_path=self.flight_data_transformation_artifact.transformed_object_file_path)


            if best_model_detail.best_score < self.model_trainer_config.expected_accuracy:
                logging.info("No best model found with score more than base score")
                raise Exception("No best model found with score more than base score")

            reg_model = RegModel(preprocessing_object=preprocessing_obj,
                                       trained_model_object=best_model_detail.best_model)
            logging.info("Created reg model object with preprocessor and model")
            logging.info("Created best model file path.")
            save_object(self.model_trainer_config.trained_reg_model_file_path , reg_model)   
            model_trainer_artifact = ModelTrainerArtifactRegression(
                trained_model_file_path=self.model_trainer_config.trained_reg_model_file_path,
                metric_artifact=metric_artifact,
            )
            logging.info(f"Regression evalution metrices:{metric_artifact}")
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise VoyageAnalyticsException(e, sys) from e
        
        
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates a model trainer steps
        
        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            reg_model_trainer_artifact = self.initiate_reg_model_trainer()
            class_model_trainer_artifact = self.initiate_classification_model_trainer()
            recumendation_model_trainer_artifact = self.initiate_recommendation_model_trainer()
            
            model_trainer_artifact = ModelTrainerArtifact(
                model_trainer_artifact_regression=reg_model_trainer_artifact,
                model_trainer_artifact_classification=class_model_trainer_artifact,
                model_trainer_artifact_recumendation=recumendation_model_trainer_artifact
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise VoyageAnalyticsException(e, sys) from e 