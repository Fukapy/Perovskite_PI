# train.py
# editted 2023/3/13

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna

def metrics(Y_train_real,Y_train_pred,Y_test_real,Y_test_pred):
    R2_train = r2_score(Y_train_real,Y_train_pred)
    RMSE_train = np.sqrt(mean_squared_error(Y_train_real,Y_train_pred))
    MAE_train = mean_absolute_error(Y_train_real,Y_train_pred)
    R2_test = r2_score(Y_test_real,Y_test_pred)
    RMSE_test = np.sqrt(mean_squared_error(Y_test_real,Y_test_pred))
    MAE_test = mean_absolute_error(Y_test_real,Y_test_pred)
    print("R2_train", R2_train)
    print("RMSE_train", RMSE_train)
    print("MAE_train", MAE_train)
    print("R2_test", R2_test)
    print("RMSE_test", RMSE_test)
    print("MAE_test", MAE_test)
    return R2_train, RMSE_train, MAE_train, R2_test, RMSE_test, MAE_test

def makemodel(model_name, max_depth, max_leaf_nodes, n_estimators, min_samples_split, min_samples_leaf, dim, n_mid, activation, solver, lr, epoch):
    
    if model_name == "RF":
        model = RandomForestRegressor(max_depth = max_depth,
                                     max_leaf_nodes = max_leaf_nodes,
                                     n_estimators = n_estimators,
                                     min_samples_split = min_samples_split,
                                     min_samples_leaf = min_samples_leaf)
        
        
    elif model_name == "GBDT":
        model = GradientBoostingRegressor(max_depth = max_depth,
                                     max_leaf_nodes = max_leaf_nodes,
                                     n_estimators = n_estimators,
                                     min_samples_split = min_samples_split,
                                     min_samples_leaf = min_samples_leaf)

    elif model_name == "NN":
        model = MLPRegressor(hidden_layer_sizes = \
                             tuple([dim for dim_count in range(n_mid)]),
                             activation = activation,
                             solver = solver,
                             learning_rate_init = lr,
                             max_iter = epoch)
        
    return model




# optuna
    
def objective_variable(model_name, X_train, y_train, valid_ratio_in_train):
    
    
    def objective(trial):
        
        if model_name == "RF":
            
            n_estimators =  trial.suggest_int("n_estimators", 10, 300, step = 10)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
            
            regr = RandomForestRegressor(n_estimators = n_estimators,
                                         min_samples_split = min_samples_split,
                                         min_samples_leaf = min_samples_leaf,
                                         n_jobs = 2,
                                         random_state = 0)
            
        elif model_name == "GBDT":
            
            n_estimators =  trial.suggest_int("n_estimators", 10, 300, step = 10)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
            lr = trial.suggest_categorical("lr", [0.5, 0.3, 0.1, 0.05, 0.03, 0.01])
            
            regr = GradientBoostingRegressor(n_estimators =n_estimators,
                                             min_samples_split =min_samples_split,
                                             min_samples_leaf = min_samples_leaf,
                                             learning_rate = lr)
            
        elif model_name == "NN":
            
            dim = trial.suggest_categorical("dim", [16,32,64,128,256])
            n_mid = trial.suggest_int("n_mid", 1, 5)
            activation = trial.suggest_categorical("activation", ["logistic", "tanh", "relu"])
            solver = trial.suggest_categorical("solver", ["sgd", "adam"])
            lr = trial.suggest_categorical("lr", [1e-3, 1e-4, 1e-5, 1e-6])
            epoch = trial.suggest_int("epoch", 100, 1000, step = 100)
            
            regr = MLPRegressor(hidden_layer_sizes = \
                             tuple([dim for dim_count in range(n_mid)]),
                             activation = activation,
                             solver = solver,
                             learning_rate_init = lr,
                             max_iter = epoch)
            
        
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size = valid_ratio_in_train, random_state = 0)
        regr.fit(X_tr, y_tr)
        y_val_pred = regr.predict(X_val)
        mse = mean_squared_error(y_val, y_val_pred)

        return mse

    return objective
    