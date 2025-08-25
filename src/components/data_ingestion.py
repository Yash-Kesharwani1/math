import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components import data_transformation, model_trainer


@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path:str = os.path.join('artifacts', 'test.csv')
    raw_data_path:str = os.path.join('artifacts', 'data.csv')


class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info('Data Ingestion Started')

            df = pd.read_csv("notebook/data/data.csv")
            
            logging.info('Data Reading Successful')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info('Train Test Split Done')

            train_set.to_csv(self.ingestion_config.train_data_path, index=False)

            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info('Data Ingestion Done')

            return (
                self.ingestion_config.train_data_path, 
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)
        

if __name__=='__main__':
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()

    obj2 = data_transformation.DataTransformation()
    train_arr, test_arr, path = obj2.initiate_data_transformation(train_path=train_path, test_path=test_path)

    obj3 = model_trainer.ModelTrainer()
    r2__score = obj3.initiate_model_trainer(train_array=train_arr, test_array=test_arr)
    print(r2__score)