'''
This library contains functions for testing the churn_library.py library
'''


import os
import logging
import pandas as pd
import numpy as np

import churn_library as cls

# Set up the logging
logger = logging.getLogger('churn_library')
logger.setLevel(logging.INFO)

# Create a file handler
fh = logging.FileHandler('logs/churn_library.log')
fh.setLevel(logging.INFO)

# Create a logging format
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(fh)



def test_import(import_data):
    '''
    Tests the import_data function - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logger.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logger.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert isinstance(
            df, pd.DataFrame), "The returned value should be a pandas DataFrame"
    except AssertionError as err:
        logger.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err

    return df


def test_eda(perform_eda, data):
    '''
    Tests the perform_eda function
    '''
    output_folder = "images/eda"
    try:
        perform_eda(data, output_folder)
        logger.info("Testing perform_eda: SUCCESS")
    except FileNotFoundError as err:
        logger.error("Testing perform_eda: The file wasn't found")
        raise err

    try:
        assert os.path.exists(
            os.path.join(
                output_folder, "churn_hist.png")), "The churn_hist.png file was not found"
    except AssertionError as err:
        logger.error(
            "Testing perform_eda: The churn_hist.png file was not found")
        raise err

    try:
        assert os.path.exists(
            os.path.join(
                output_folder, "age_hist.png")), "The age_hist.png file was not found"
    except AssertionError as err:
        logger.error(
            "Testing perform_eda: The age_hist.png file was not found")
        raise err

    try:
        assert os.path.exists(
            os.path.join(
                output_folder, "marital_hist.png")), "The marital_hist.png file was not found"
    except AssertionError as err:
        logger.error(
            "Testing perform_eda: The marital_hist.png file was not found")
        raise err

    try:
        assert os.path.exists(
            os.path.join(
                output_folder, "corr_matrix.png")), "The corr_matrix.png file was not found"
    except AssertionError as err:
        logger.error(
            "Testing perform_eda: The corr_matrix.png file was not found")
        raise err


def test_encoder_helper(encoder_helper, data):
    '''
    Tests the encoder_helper function
    '''
    try:
        encoded_df = encoder_helper(
            data, data.select_dtypes(
                include='object').columns, "Churn")
        logger.info("Testing encoder_helper: SUCCESS")
    except FileNotFoundError as err:
        logger.error("Testing encoder_helper: The file wasn't found")
        raise err

    try:
        assert "Gender_churn_rate" in encoded_df.columns
    except AssertionError as err:
        logger.error(
            "Testing encoder_helper failed: The Gender_churn_rate column was not found")
        raise err

    try:
        assert "Education_Level_churn_rate" in encoded_df.columns
    except AssertionError as err:
        logger.error(
            "Testing encoder_helper failed: The Education_Level_churn_rate column was not found")
        raise err

    try:
        assert encoded_df['Gender_churn_rate'].dtype == float
    except AssertionError as err:
        logger.error(
            "Testing encoder_helper failed: The gender_churn_rate column is not of type float")
        raise err

    return encoded_df


def test_perform_feature_engineering(perform_feature_engineering, data):
    '''
    Test perform_feature_engineering function.
    '''
    try:
        X_train, X_test, y_train, y_test = perform_feature_engineering(
            data, "Churn")
        logger.info("Testing perform_feature_engineering: SUCCESS")
    except FileNotFoundError as err:
        logger.error(
            "Testing perform_feature_engineering: The file wasn't found")
        raise err

    try:
        assert isinstance(
            X_train, pd.DataFrame), "X_train should be a pandas DataFrame"
    except AssertionError as err:
        logger.error(
            "Testing perform_feature_engineering failed: X_train is not a pandas DataFrame")
        raise err

    try:
        assert isinstance(
            X_test, pd.DataFrame), "X_test should be a pandas DataFrame"
    except AssertionError as err:
        logger.error(
            "Testing perform_feature_engineering failed: X_test is not a pandas DataFrame")
        raise err

    try:
        assert X_train.shape[1] == X_test.shape[1], "X_train and X_test should have the same number of columns"
    except AssertionError as err:
        logger.error(
            "Testing perform_feature_engineering failed: X_train and X_test do not have the same number of columns")
        raise err

    return X_train, X_test, y_train, y_test


def test_train_models(train_models, X_train, X_test, y_train, y_test):
    '''
    Test train_models function.
    '''
    output_model = "models"
    output_results = "images/results"
    try:
        train_models(
            X_train,
            X_test,
            y_train,
            y_test,
            output_model,
            output_results)
        logger.info("Testing train_models: SUCCESS")
    except FileNotFoundError as err:
        logger.error("Testing train_models: The file wasn't found")
        raise err

    try:
        assert os.path.exists(
            os.path.join(
                output_model,
                "logistic_model.pkl"))
    except AssertionError as err:
        logger.error(
            "Testing train_models failed: The logistic_model.pkl file was not found")
        raise err

    try:
        assert os.path.exists(
            os.path.join(
                output_model,
                "random_forest_model.pkl"))
    except AssertionError as err:
        logger.error(
            "Testing train_models failed: The random_forest_model.pkl file was not found")
        raise err


if __name__ == "__main__":
    DATA = test_import(cls.import_data)
    test_eda(cls.perform_eda, DATA)
    DATA = test_encoder_helper(cls.encoder_helper, DATA)
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = test_perform_feature_engineering(
        cls.perform_feature_engineering, DATA)
    test_train_models(cls.train_models, X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)
    logger.info("All tests passed successfully!")
