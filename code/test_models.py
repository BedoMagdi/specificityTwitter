import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
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

    logging.info(f"RBFSVR_model mae = {mae}\n")

    return mae, y_predict


def tune_random_forest_model(X_train, y_train):
    # Start the timer
    start_time = time.time()
    
    # Define the hyperparameter grid
    param_grid = {
        'n_estimators': [10, 50, 100, 200],  # List of values for number of estimators
        'max_depth': [None, 10, 20, 30],  # List of values for maximum depth
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
    logging.info("Time taken for Grid Search in tune_random_forest_model: {:.2f} seconds".format(time_taken))
    

    # Get the best hyperparameter values
    best_n_estimators = grid_search.best_params_['n_estimators']
    best_max_depth = grid_search.best_params_['max_depth']

    logging.info(f"random_forest_model params -> best_n_estimators = {best_n_estimators}, best_max_depth = {best_max_depth}\n")

    return best_n_estimators, best_max_depth


def random_forest_model(X_train, y_train, X_test, y_test, _n_estimators, _max_depth):
    regr = RandomForestRegressor(n_estimators=_n_estimators, max_depth=_max_depth, random_state=42)
    # Fit the model on the training data
    regr.fit(X_train, y_train)

    # Predict on the test data
    y_predict = regr.predict(X_test)

    # Calculate Mean Absolute Error (MAE)
    mae = round(mean_absolute_error(y_test, y_predict), 3)
    logging.info(f"random_forest_model mae = {mae}\n")

    return mae, y_predict



if __name__ == "__main__":
    ## for logging 
    out_dir = os.path.join(os.getcwd(), 'logs')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True )

    Log_Format = "%(message)s"
    logging.basicConfig(filename = f"{out_dir}/models-emotion_features.log", 
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


    ## training and testing on all features
    df_base_train = df_all.loc[:6266, 'sentlen'].values.reshape(-1, 1)
    df_base_test = df_all.loc[6267:, 'sentlen'].values.reshape(-1, 1)
    df_train_val = df_all.iloc[:6267, 3:]
    df_test = df_all.iloc[6267:, 3:]


    df_scores= pd.read_csv(args.scorefile, sep='\t', skip_blank_lines=True)


    ## for the bar graph
    # df_scores_all = df_scores.loc[:, 'Score']
    # value_counts = df_scores_all.value_counts()
    # scores = df_scores_all.values

    # plt.hist(scores, bins=30, edgecolor='black', linewidth=0.5)
    # plt.xlabel('Score')
    # plt.ylabel('Count')
    # plt.xticks(fontsize=8)
    # plt.yticks(fontsize=8)
    # #plt.show()   
    # plt.savefig('Specificity_frequency_dist.png') 

    
    # Get 'Score' column values for df_train
    base_train_scores = df_scores.loc[:6266, 'Score']
    base_test_scores = df_scores.loc[6267:, 'Score']
    train_val_scores = df_scores.loc[:6266, 'Score']
    test_scores = df_scores.loc[6267:, 'Score']
    


    ## surface and lexical only
    surface_lexical_features = ['avgwordlen', 'sentlen', 'numsymbols', 'numcapltrs', 'numnumbers', 'DT', 'NN', "VB", 'JJ', 'IN', '.', 'PRP', 'NNP', 'WP', 'ORGANIZATION', "PERCENT", 'PERSON', 'DATE', 'MONEY',
                      'TIME', 'LOCATION', 'Concrete', 'syllable_count']
    surface_lexical_features_df = df_all[surface_lexical_features]
    df_train_val_surface_lexical_features = surface_lexical_features_df.iloc[:6267, :]
    df_test_surface_lexical_features = surface_lexical_features_df.iloc[6267:, :]

    ## Emotion only
    emotion_features = ['numemoji', 'Negative', 'Positive']
    emotion_features_df = df_all[emotion_features]
    df_train_val_emotion_features = emotion_features_df.iloc[:6267,:]
    df_test_emotion_features = emotion_features_df.iloc[6267:,:]


    ## Deixis only
    deixis_features = ['personalDeixis', 'tmpDeixis', 'spatialDeixis']
    deixis_features_df = df_all[deixis_features]
    df_train_val_deixis_features = deixis_features_df.iloc[:6267,:]
    df_test_deixis_features = deixis_features_df.iloc[6267:,:]


    ## Distributional word representations only
    col_to_drop = surface_lexical_features + emotion_features + deixis_features
    embedding_df = df_all.drop(columns=col_to_drop)
    df_train_val_embedding_features = embedding_df.iloc[:6267,3:]
    df_test_embedding_features = embedding_df.iloc[6267:,3:]

    

    ## All - surface and lexical
    all_except_surface_lexical_df = df_all.drop(columns=surface_lexical_features)
    df_train_val_all_except_surface_lexical = all_except_surface_lexical_df.iloc[:6267,3:]
    df_test_all_except_surface_lexical = all_except_surface_lexical_df.iloc[6267:,3:]

    ## All - Emotion
    all_except_emotion_df = df_all.drop(columns=emotion_features)
    df_train_val_all_except_emotion = all_except_emotion_df.iloc[:6267,3:]
    df_test_all_except_emotion = all_except_emotion_df.iloc[6267:,3:]


    ## All - Deixis
    all_except_deixis_df = df_all.drop(columns=deixis_features)
    df_train_val_all_except_deixis= all_except_deixis_df.iloc[:6267,3:]
    df_test_all_except_deixis = all_except_deixis_df.iloc[6267:,3:]
    

    ## All - Distributional word representations
    embedding_columns_removed = list(filter(lambda col: col not in embedding_df.columns, df_all.columns)) #set(df_all.columns) - set(embedding_df.columns)
    all_except_embedding_df = df_all[embedding_columns_removed]
    df_train_val_all_except_embedding= all_except_embedding_df.iloc[:6267,:]
    df_test_all_except_embedding = all_except_embedding_df.iloc[6267:,:]

    

    ## try: surface and lexical + Distributional word representations
    cols = list(embedding_df.columns) + surface_lexical_features
    cols_df = df_all[cols]
    df_train_val_cols = cols_df.iloc[:6267,3:]
    df_test_cols = cols_df.iloc[6267:,3:]


    ## RF
    #print('******** Random Forest ********')
    #best_n_estimators, best_max_depth, best_max_features = tune_random_forest_model(df_train_val, train_val_scores)
    # best_n_estimators = 200 
    # best_max_depth = 20
    # mae, y_predict = random_forest_model(df_train_val, train_val_scores, df_test, test_scores, best_n_estimators, best_max_depth)

    # plt.hist(y_predict, bins=30, edgecolor='black', linewidth=0.5)
    # plt.xlabel('Score')
    # plt.ylabel('Count')
    # plt.xticks(fontsize=8)
    # plt.yticks(fontsize=8)
    # plt.xlim(1.0, 5.0)
    # #plt.show()   
    # plt.savefig('RF_Specificity_frequency_dist.png') 


    # corr, _ = pearsonr(y_predict, test_scores)
    # print('Pearsons correlation: %.3f' % corr)
    # print("mae model", mae)

    # logging.info(f"random_forest_model base:\n")
    # mae, y_predict = random_forest_model(df_base_train, base_train_scores, df_base_test, base_test_scores, best_n_estimators, best_max_depth)
    # corr, _ = pearsonr(y_predict, base_test_scores)
    # print('Pearsons correlation: %.3f' % corr)
    # print("mae base", mae)


    ## RBFSVR_model
    # print('******** RBFSVR_model********')
    # #best_C, best_gamma = tune_RBFSVR_model(df_train_val, train_val_scores)
    best_C = 1
    best_gamma = 0.01
    mae, y_predict = RBFSVR_model(df_train_val, train_val_scores, df_test, test_scores, best_C, best_gamma)

    plt.hist(y_predict, bins=30, edgecolor='black', linewidth=0.5)
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.xlim(1.0, 5.0)
    #plt.show()   
    plt.savefig('SVR_Specificity_frequency_dist.png') 


    # corr, _ = pearsonr(y_predict, base_test_scores)
    # print('Pearsons correlation: %.3f' % corr)
    # print("mae base", mae)

    # logging.info(f"RBFSVR_model base:\n")
    # mae, y_predict = RBFSVR_model(df_base_train, base_train_scores, df_base_test, base_test_scores, best_C, best_gamma)
    # corr, _ = pearsonr(y_predict, base_test_scores)
    # print('Pearsons correlation: %.3f' % corr)
    # print("mae base", mae)



    