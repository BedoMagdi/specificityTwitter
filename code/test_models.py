import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import argparse
import time, logging
import os


def tune_RBFSVR_model(X_train, y_train):
    # Start the timer
    start_time = time.time()
    
    # Define the hyperparameter grid for grid search
    param_grid = {'C': [0.01, 0.1, 1, 10], 'gamma': [0.01, 0.1, 1, 10]}
    
    # Create an SVR model
    regr = SVR(kernel="rbf")

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(regr, param_grid, scoring='neg_mean_absolute_error', cv=5)
    grid_search.fit(X_train, y_train)

    # Calculate the time taken
    end_time = time.time()
    time_taken = end_time - start_time
    print("Time taken for Grid Search in tune_RBFSVR_model: {:.2f} seconds".format(time_taken))
    logging.info("Time taken for Grid Search in tune_RBFSVR_model: {:.2f} seconds".format(time_taken))

    # Get the best hyperparameters
    best_C = grid_search.best_params_['C']
    best_gamma = grid_search.best_params_['gamma']

    logging.info(f"RBFSVR_model params -> best_C = {best_C}, best_gamma = {best_gamma}")

    return best_C, best_gamma


def RBFSVR_model(X_train, y_train, X_test, y_test, c, gamma):
    model = SVR(kernel="rbf", C=c, gamma=gamma)
    model.fit(X_train, y_train)

    ## Predict 
    y_predict = model.predict(X_test)

    ## Calculate Mean Absolute Error (MAE)
    mae = round(mean_absolute_error(y_test, y_predict), 3)

    logging.info(f"RBFSVR_model mae = {mae}")

    return mae, y_predict


def tune_random_forest_model(X_train, y_train):
    # Start the timer
    start_time = time.time()
    
    # Define the hyperparameter grid
    param_grid = {
        'n_estimators': [10, 50, 100, 200],  # List of values for number of estimators
        'max_depth': [None, 10, 20, 30]  # List of values for maximum depth
    }

    # Create a Random Forest Regression model
    regr = RandomForestRegressor(random_state=42)

    # Perform grid search
    grid_search = GridSearchCV(regr, param_grid, scoring='neg_mean_absolute_error', cv=5)
    grid_search.fit(X_train, y_train)

    # Calculate the time taken
    end_time = time.time()
    time_taken = end_time - start_time
    print("Time taken for Grid Search in tune_random_forest_model: {:.2f} seconds".format(time_taken))

    # Get the best hyperparameter values
    best_n_estimators = grid_search.best_params_['n_estimators']
    best_max_depth = grid_search.best_params_['max_depth']

    return best_n_estimators, best_max_depth


def random_forest_model(X_train, y_train, X_test, y_test, _n_estimators, _max_depth):
    regr = RandomForestRegressor(n_estimators=_n_estimators, max_depth=_max_depth, random_state=42)
    # Fit the model on the training data
    regr.fit(X_train, y_train)

    # Predict on the test data
    y_predict = regr.predict(X_test)

    # Calculate Mean Absolute Error (MAE)
    mae = round(mean_absolute_error(y_test, y_predict), 3)

    return mae, y_predict



if __name__ == "__main__":
    ## for logging 
    out_dir = os.path.join(os.getcwd(), 'logs')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True )

    Log_Format = "%(levelname)s %(asctime)s - %(message)s"
    logging.basicConfig(filename = f"{out_dir}/models.log", 
                    filemode = "w",
                    format = Log_Format, 
                    level = logging.INFO)


    argparser = argparse.ArgumentParser()
    argparser.add_argument("--inputfile",
                           help="input raw text file, one sentence per line, tokenized",
                           required=True)
    
    argparser.add_argument("--scorefile",
                        help="input raw text file, one sentence per line, tokenized",
                        required=True)
    
    args = argparser.parse_args()

    df_all = pd.read_csv(args.inputfile, skip_blank_lines=True)


    df_base_train = df_all.loc[:6266, 'sentlen'].values.reshape(-1, 1)
    df_base_test = df_all.loc[6267:, 'sentlen'].values.reshape(-1, 1)
    df_train_val = df_all.iloc[:6267, 3:]
    df_test = df_all.iloc[6267:, 3:]

    print(df_test.shape)

    df_scores= pd.read_csv(args.scorefile, sep='\t', skip_blank_lines=True)
    
    # Get 'Score' column values for df_train
    base_train_scores = df_scores.loc[:6266, 'Score']
    base_test_scores = df_scores.loc[6267:, 'Score']
    train_val_scores = df_scores.loc[:6266, 'Score']
    test_scores = df_scores.loc[6267:, 'Score']
    
    print(test_scores.shape)

    best_C, best_gamma = tune_RBFSVR_model(df_train_val, train_val_scores)
    mae, y_predict = RBFSVR_model(df_train_val, train_val_scores, df_test, test_scores, best_C, best_gamma)
    print("mae model", mae)
    
    logging.info(f"RBFSVR_model base:\n")
    mae, y_predict = RBFSVR_model(df_base_train, base_train_scores, df_base_test, base_test_scores, best_C, best_gamma)
    print("mae base", mae)
