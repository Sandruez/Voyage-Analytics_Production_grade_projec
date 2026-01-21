import os
import sys

import numpy as np
import dill
import yaml
import pandas as pd
from pandas import DataFrame
from voyage_analytics.exception import VoyageAnalyticsException
from voyage_analytics.logger import logging
import pandas as pd
# from voyage_analytics.entity.artifact_entity import ModelTrainerArtifact 

class Local_Estimator_Class:
    def __init__(self,):
        try:
            self.trained_reg_model_file_path='artifact/01_21_2026_02_32_38/model_trainer/trained_model/regression/model.pkl'
            self.trained_class_model_file_path='artifact/01_21_2026_02_32_38/model_trainer/trained_model/classification/model.pkl'
            self.trained_recumend_model_file_path='artifact/01_21_2026_02_32_38/model_trainer/trained_model/recumendation/model.pkl'
        except Exception as e:
            raise VoyageAnalyticsException(e, sys) from e
    
    @staticmethod  
    def load_object(file_path: str) -> object:
        logging.info("Entered the load_object method of utils")

        try:
            with open(file_path, "rb") as file_obj:
                obj = dill.load(file_obj)

            logging.info("Exited the load_object method of utils")

            return obj

        except Exception as e:
            raise VoyageAnalyticsException(e, sys) from e
    
    
    def regression_predict_func(self,test_df:DataFrame)->object:
        try:
            
            logging.info("Initialisation of reg_predict_func for flight price predictions")
            logging.info(f"Loading model object from {self.trained_reg_model_file_path}")
            
            model_obj=self.load_object(self.trained_reg_model_file_path)
            # return a object type value..
            preds=model_obj.predict(dataframe=test_df)
            return preds
        
        except Exception as e:
            raise VoyageAnalyticsException(e, sys) from e
             
    def classification_predict_func(self,names_df:DataFrame)->DataFrame:
        try:
            
            logging.info("Initialisation of reg_predict_func for flight price predictions")
            logging.info(f"Loading model object from {self.trained_class_model_file_path}")
            
            model_obj=self.load_object(self.trained_class_model_file_path)
            
            #return a object type value.'.
            preds=model_obj.predict(names_df)
            return preds
        
        except Exception as e:
            raise VoyageAnalyticsException(e, sys) from e
        
    
    def recommendation_predict_func(self,hotel_name:str)->DataFrame:
        try:
            
            logging.info("Initialisation of reg_predict_func for flight price predictions")
            logging.info(f"Loading model object from {self.trained_recumend_model_file_path}")
            
            model_obj=self.load_object(self.trained_recumend_model_file_path)
            
            #return a object type value..
            preds=model_obj.predict(hotel_name)
            return preds
        
        except Exception as e:
            raise VoyageAnalyticsException(e, sys) from e
        
        
    