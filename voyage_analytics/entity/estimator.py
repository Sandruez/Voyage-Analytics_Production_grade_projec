import sys

from pandas import DataFrame
from sklearn.compose  import ColumnTransformer
from voyage_analytics.exception import VoyageAnalyticsException
from voyage_analytics.logger import logging


class RegModel:
    def __init__(self, preprocessing_object: ColumnTransformer, trained_model_object: object):
        """
        :param preprocessing_object: Input Object of preprocesser
        :param trained_model_object: Input Object of trained model 
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, dataframe: DataFrame) -> object:
        """
        Function accepts raw inputs and then transformed raw input using preprocessing_object
        which guarantees that the inputs are in the same format as the training data
        At last it performs prediction on transformed features
        """
        logging.info("Entered predict method of UTruckModel class")

        try:
            logging.info("Using the trained model to get predictions")

            transformed_feature = self.preprocessing_object.transform(dataframe)

            logging.info("Used the trained model to get predictions")
            return self.trained_model_object.predict(transformed_feature)

        except Exception as e:
            raise VoyageAnalyticsException(e, sys) from e

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"
    
    
class ClassificationModel:
    def __init__(self,trained_model_object: object):
        """
        :param preprocessing_object: Input Object of preprocesser
        :param trained_model_object: Input Object of trained model 
        """
        self.trained_model_object = trained_model_object

    def predict(self, dataframe: DataFrame) -> DataFrame:
        """
        Function accepts raw inputs and then transformed raw input using preprocessing_object
        which guarantees that the inputs are in the same format as the training data
        At last it performs prediction on transformed features
        """
        logging.info("Entered predict method of ClassificationModel class")

        try:
            logging.info("Using the trained model to get predictions")

            logging.info("Used the trained model to get predictions")
            return self.trained_model_object.predict(dataframe.iloc[:,0])

        except Exception as e:
            raise VoyageAnalyticsException(e, sys) from e

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"
    
    
class RecumendationModel:
    def __init__(self,trained_model_object: object, hotel_profiles: DataFrame, hotel_features_matrix: np.ndarrays):
        """
        :param preprocessing_object: Input Object of preprocesser
        :param trained_model_object: Input Object of trained model 
        """
        self.trained_model_object = trained_model_object
        self.hotel_profiles = hotel_profiles
        self.hotel_features_matrix = hotel_features_matrix

    def predict(self, hotel_name: str) -> DataFrame:
        """
        it performs prediction
        """
        logging.info("Entered predict method of RecumendationModel class")

        try:
                    # 1. Find the index of the hotel
            try:
                idx = self.hotel_profiles[self.hotel_profiles['name'] == hotel_name].index[0]
            except Exception as e:
                raise VoyageAnalyticsException(e, sys) from e

            # 2. Find nearest neighbors
            distances, indices = self.trained_model_object.kneighbors([self.hotel_features_matrix[idx]])

            # 3. Return results (Skip first one because it's the hotel itself)
            similar_indices = indices[0][1:]

            return self.hotel_profiles.iloc[similar_indices][['name', 'place', 'price', 'popularity']]
            logging.info("Using the trained model to get predictions")
        
        except Exception as e:
            raise VoyageAnalyticsException(e, sys) from e
        
        
   