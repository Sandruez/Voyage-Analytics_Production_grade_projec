import os
import sys

from pandas import DataFrame
from sklearn.model_selection import train_test_split

from voyage_analytics.entity.config_entity import DataIngestionConfig
from voyage_analytics.entity.artifact_entity import DataIngestionArtifact, DataIngestionArtifactUsers, DataIngestionArtifactFlights, DataIngestionArtifactHotels
from voyage_analytics.exception import VoyageAnalyticsException
from voyage_analytics.logger import logging
from voyage_analytics.data_access.voyage_data import VoyageData

class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig=DataIngestionConfig()):
        """
        :param data_ingestion_config: configuration for data ingestion
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise VoyageAnalyticsException(e,sys)
        



    def export_data_into_feature_store(self)->tuple[DataFrame, DataFrame, DataFrame]:
        """
        Method Name :   export_data_into_feature_store
        Description :   This method exports data from mongodb to csv file
        
        Output      :   data is returned as artifact of data ingestion components
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            #Users data export
            logging.info(f"Exporting Users data from mongodb")
            voyage_data = VoyageData()
            users_dataframe = voyage_data.export_collection_as_dataframe(collection_name=
                                                                   self.data_ingestion_config.users_collection_name)
            logging.info(f"Shape of users dataframe: {users_dataframe.shape}")
            users_feature_store_file_path  = self.data_ingestion_config.users_feature_store_file_path
            dir_path = os.path.dirname(users_feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            logging.info(f"Saving exported users data into feature store file path: {users_feature_store_file_path}")
            users_dataframe.to_csv(users_feature_store_file_path,index=False,header=True)
            logging.info(f"Saved exported users data into feature store file path: {users_feature_store_file_path}")
            
            #flights data export
            logging.info(f"Exporting flights data from mongodb")
            voyage_data = VoyageData()      
            flights_dataframe = voyage_data.export_collection_as_dataframe(collection_name=
                                                                   self.data_ingestion_config.flights_collection_name)
            logging.info(f"Shape of flights dataframe: {flights_dataframe.shape}")
            flights_feature_store_file_path  = self.data_ingestion_config.flights_feature_store_file_path
            dir_path = os.path.dirname(flights_feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            logging.info(f"Saving exported flights data into feature store file path: {flights_feature_store_file_path}")
            flights_dataframe.to_csv(flights_feature_store_file_path,index=False,header=True)
            logging.info(f"Saved exported flights data into feature store file path: {flights_feature_store_file_path}")
            
            #hotels data export
            logging.info(f"Exporting hotels data from mongodb")
            voyage_data = VoyageData()
            hotels_dataframe = voyage_data.export_collection_as_dataframe(collection_name=
                                                                   self.data_ingestion_config.hotels_collection_name)
            logging.info(f"Shape of hotels dataframe: {hotels_dataframe.shape}")
            hotels_feature_store_file_path  = self.data_ingestion_config.hotels_feature_store_file_path
            dir_path = os.path.dirname(hotels_feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            logging.info(f"Saving exported hotels data into feature store file path: {hotels_feature_store_file_path}")
            hotels_dataframe.to_csv(hotels_feature_store_file_path,index=False,header=True)
            logging.info(f"Saved exported hotels data into feature store file path: {hotels_feature_store_file_path}")
        
            return users_dataframe, flights_dataframe, hotels_dataframe

        except Exception as e:
            raise VoyageAnalyticsException(e,sys)
        

    def split_data_as_train_test(self,dataframe: DataFrame,dataframe_name: str,training_file_path: str,testing_file_path: str) ->None:
        """
        Method Name :   split_data_as_train_test
        Description :   This method splits the dataframe into train set and test set based on split ratio 
        
        Output      :   Folder is created in s3 bucket
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered split_data_as_train_test method of Data_Ingestion class")

        try:
            train_set, test_set = train_test_split(dataframe, test_size=self.data_ingestion_config.train_test_split_ratio)
            logging.info(f"Performed train test split on the : {dataframe_name} dataframe")
            logging.info(
                "Exited split_data_as_train_test method of Data_Ingestion class"
            )
            dir_path = os.path.dirname(training_file_path)
            os.makedirs(dir_path,exist_ok=True)
            
            logging.info(f"Exporting {dataframe_name} train and test file path.")
            train_set.to_csv(training_file_path,index=False,header=True)
            test_set.to_csv(testing_file_path,index=False,header=True)

            logging.info(f"Exported {dataframe_name} train and test file path.")
        except Exception as e:
            raise VoyageAnalyticsException(e, sys) from e
        




    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Method Name :   initiate_data_ingestion
        Description :   This method initiates the data ingestion components of training pipeline 
        
        Output      :   train set and test set are returned as the artifacts of data ingestion components
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered initiate_data_ingestion method of Data_Ingestion class")

        try:
            users_dataframe, flights_dataframe, hotels_dataframe = self.export_data_into_feature_store()

            logging.info("Got the data from mongodb")

            self.split_data_as_train_test(users_dataframe,"users",self.data_ingestion_config.class_training_file_path,self.data_ingestion_config.class_testing_file_path)
            self.split_data_as_train_test(flights_dataframe,"flights",self.data_ingestion_config.reg_training_file_path,self.data_ingestion_config.reg_testing_file_path)
            self.split_data_as_train_test(hotels_dataframe,"hotels",self.data_ingestion_config.recumend_training_file_path,self.data_ingestion_config.recumend_testing_file_path)
            logging.info("Performed train test split on the datasets")

            logging.info(
                "Exited initiate_data_ingestion method of Data_Ingestion class"
            )

            user_data_ingestion_artifact = DataIngestionArtifactUsers(trained_file_path=self.data_ingestion_config.class_training_file_path,
            test_file_path=self.data_ingestion_config.class_testing_file_path)
            logging.info(f"User Data ingestion artifact: {user_data_ingestion_artifact}")
            
            flight_data_ingestion_artifact = DataIngestionArtifactFlights(trained_file_path=self.data_ingestion_config.reg_training_file_path,
            test_file_path=self.data_ingestion_config.reg_testing_file_path)
            logging.info(f"Flight Data ingestion artifact: {flight_data_ingestion_artifact}")   
            
            hotel_data_ingestion_artifact = DataIngestionArtifactHotels(trained_file_path=self.data_ingestion_config.recumend_training_file_path,
            test_file_path=self.data_ingestion_config.recumend_testing_file_path) 
            logging.info(f"Hotel Data ingestion artifact: {hotel_data_ingestion_artifact}")

            data_ingestion_artifact = DataIngestionArtifact(
                user_data_ingestion_artifact=user_data_ingestion_artifact,
                flight_data_ingestion_artifact=flight_data_ingestion_artifact,
                hotel_data_ingestion_artifact=hotel_data_ingestion_artifact
            )
            return data_ingestion_artifact

        except Exception as e:
            raise VoyageAnalyticsException(e, sys) from e
