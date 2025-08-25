import os , sys
from dataclasses import dataclass

# sklearn
# Ensemble Algos
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
# Linear
from sklearn.linear_model import LinearRegression
# Neighbors
from sklearn.neighbors import KNeighborsRegressor
# Tree
from sklearn.tree import DecisionTreeRegressor
# Metrics
from sklearn.metrics import r2_score


# User defined class
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Model Training Started')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1], train_array[:,-1],
                test_array[:,:-1], test_array[:,-1]
            )

            models = {
                'Random Forest': RandomForestRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'Linear Regression': LinearRegression(),
                'AdaBoost Regressor': AdaBoostRegressor()
            }

            params = {
                'Decision Tree':{
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
                    # 'splitter': ['best', 'random'],
                    # 'max_features':['sqrt', 'log2']
                },
                'Random Forest':{
                    # 'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'possion'],
                    # 'max_features': ['sqrt', 'log2', 'None'],
                    'n_estimators': [8, 16, 32, 64, 126, 256]
                },
                'Gradient Boosting': {
                    # 'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'possion'],
                    # 'max_features': ['sqrt', 'log2', 'Auto'],
                    'learning_rate' : [0.1, 0.01, 0.05],
                    # 'loss' : ['squared_error', 'huber', 'absolute_error', 'quantile'],
                    # 'n_estimator' : [8, 16, 32, 64, 128]
                },
                'Linear Regression' : {},
                'AdaBoost Regressor' : {
                    'learning_rate': [0.1, 0.01, 0.05],
                    # 'loss' : ['linear', 'square', 'exponential'], 
                    # 'n_estimator' : [8, 16, 32, 64, 128]
                }
            }

            logging.info('Model Evaluation Start')
            model_report : dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param=params)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            logging.info(f'Best Model name is {best_model_name}')

            if best_model_score < 0.6 : 
                raise CustomException('No best model found.')
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square
        
        except Exception as e:
            raise CustomException(e, sys)

