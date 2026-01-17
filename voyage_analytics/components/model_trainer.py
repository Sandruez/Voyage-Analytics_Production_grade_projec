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
from voyage_analytics.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact,RegressionMetricArtifact,ClassificationMetricArtifact,RecumendationMetricArtifact
from voyage_analytics.entity.estimator import RegModel
from voyage_analytics.constants import TARGET_COLUMN_USERS

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
    #     try:
    
    #         transformed_object_file_path:str 
    # transformed_train_file_path:str
    # transformed_test_file_path:str

            x_train, y_train, x_test, y_test = train_df.drop(columns=[TARGET_COLUMN_USERS], axis=1), train_df[TARGET_COLUMN_USERS], test_df.drop(columns=[TARGET_COLUMN_USERS], axis=1), test_df[TARGET_COLUMN_USERS]

            model = make_pipeline(
            CountVectorizer(analyzer='char_wb', ngram_range=(2, 4)),
            MultinomialNB()
            )
            logging.info("initiating training the classification model")
            model.fit(x_train, y_train)
            model_obj = model
            logging.info("trained the classification model")

            y_pred = model_obj.predict(x_test)
            metric_artifact = ClassificationMetricArtifact(
                accuracy_score=accuracy_score(y_true=y_test, y_pred=y_pred),
                precision_score=precision_score(y_true=y_test, y_pred=y_pred),
                recall_score=recall_score(y_true=y_test, y_pred=y_pred),
                f1_score=f1_score(y_true=y_test, y_pred=y_pred),
            )
            return model_obj, metric_artifact
        except Exception as e:
            raise VoyageAnalyticsException(e, sys) from e

    def get_recommendation_model_object_and_report(self, train: np.array, test: np.array) -> Tuple[object, object]:
        """
        Method Name :   get_recommendation_model_object_and_report
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
            metric_artifact = RecumendationMetricArtifact(
                recall=recall_score(y_true=y_test, y_pred=y_pred),
                precision=precision_score(y_true=y_test, y_pred=y_pred),
                f1-score=f1_score(y_true=y_test, y_pred=y_pred),
            )
            return best_model_detail, metric_artifact
        
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
            train_arr = load_csv_data(file_path=self.user_data_transformation_artifact.transformed_train_file_path)
            test_arr = load_csv_data(file_path=self.user_data_transformation_artifact.transformed_test_file_path) 
            
            best_model ,metric_artifact = self.get_classification_model_object_and_report(train=train_arr, test=test_arr)
            preprocessing_obj = load_object(file_path=self.user_data_transformation_artifact.transformed_object_file_path)
            
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
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise VoyageAnalyticsException(e, sys) from e   
        

   def initiate_recommendation_model_trainer(self, ) -> ModelTrainerArtifactRecommendation:
        logging.info("Entered initiate_recommendation_model_trainer method of ModelTrainer class")
        """
        Method Name :   initiate_recommendation_model_trainer
        Description :   This function initiates a model trainer steps
        
        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            train_arr = load_csv_data(file_path=self.user_data_transformation_artifact.transformed_train_file_path)
            test_arr = load_csv_data(file_path=self.user_data_transformation_artifact.transformed_test_file_path) 
            
            best_model ,metric_artifact = self.get_classification_model_object_and_report(train=train_arr, test=test_arr)
            preprocessing_obj = load_object(file_path=self.user_data_transformation_artifact.transformed_object_file_path)
            
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
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise USvisaException(e, sys) from e