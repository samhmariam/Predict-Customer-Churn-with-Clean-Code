# library doc string
'''
This library contains functions for importing data, performing EDA, feature engineering, and training models

'''

# import libraries

import os
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, RocCurveDisplay
from sklearn.model_selection import GridSearchCV

import joblib

# Set the QT_QPA_PLATFORM environment variable to 'offscreen' to prevent
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def import_data(pth):
    '''
    Returns dataframe for the csv found at the specified path.

    input:
        pth: Path to the csv file.
    output:
        df: Pandas dataframe
    '''

    data = pd.read_csv(pth)
    return data


def perform_eda(data_eda, output_folder="images/eda"):
    '''
    Perform Exploratory Data Analysis (EDA) on dataframe and save figures to images folder
    input:
        df: Pandas dataframe.

    output:
        Path to the folder for storing generated figures
    '''

    # Create a churn column from the Attrition_Flag column
    data_eda['Churn'] = data_eda['Attrition_Flag'].apply(
        lambda x: 1 if x == 'Attrited Customer' else 0)
    # Drop the original Attrition_Flag column
    data_eda.drop('Attrition_Flag', axis=1, inplace=True)
    # Create a histogram of the Churn column and save it to the images folder
    plt.figure(figsize=(10, 6))
    sns.histplot(data_eda['Churn'], kde=False)
    plt.title('Churn Distribution')
    plt.xlabel('Churn')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_folder, "churn_hist.png"))
    plt.close()
    # Create a histogram of the Customer_Age column and save it to the images
    # folder
    plt.figure(figsize=(10, 6))
    sns.histplot(data_eda['Customer_Age'], kde=False)
    plt.title('Customer Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_folder, "age_hist.png"))
    plt.close()
    # Create a histogram of the Marital_Status column and save it to the
    # images folder
    plt.figure(figsize=(10, 6))
    sns.histplot(data_eda['Marital_Status'], kde=False)
    plt.title('Marital Status Distribution')
    plt.xlabel('Marital Status')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_folder, "marital_hist.png"))
    plt.close()
    # Create a correlation matrix of the data and save it to the images folder
    plt.figure(figsize=(10, 6))
    sns.heatmap(data_eda[data_eda.select_dtypes(
        include=np.number).columns].corr(), annot=True)
    plt.title('Correlation Matrix')
    plt.savefig(os.path.join(output_folder, "corr_matrix.png"))
    plt.close()


def encoder_helper(df, category_lst, response):
    '''
    Ecodes categorical columns into new columns containing the proportion of churn
    for each category.

    input:
            df: Pandas dataframe
            category_lst: List of columns that contain categorical features
            response: Name of the response variable (optional).

    output:
            df: Modified pandas dataframe with encoded columns.
    '''
    for category in category_lst:
        churn_rate = df.groupby(category)[response].mean()
        df[f"{category}_churn_rate"] = df[category].apply(
            lambda x: churn_rate.loc[x])
    return df


def perform_feature_engineering(data, response):
    '''
    Performs feature engineering on the dataframe, including encoding and feature
    selection.
    input:
              df: Pandas dataframe
              response: Name of the response variable.

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    # Separate the features (X) and target variable (y)
    X = data.drop(response, axis=1)
    y = data[response]

    # Encode the categorical columns
    cat_cols = X.select_dtypes(include='object').columns
    data = encoder_helper(data, cat_cols, response)

    # Drop the original categorical columns and the response variable
    X = data.drop([response] + list(cat_cols), axis=1)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf,
                                output_folder="images"):
    '''
    Generates classification reports for training and testing of both models (logistic regression and random forest)
    and saves them as images in the specified output folder (defaults to "images").

    input:
            y_train: Training target values
            y_test:  Test target values
            y_train_preds_lr: Training predictions from logistic regression
            y_train_preds_rf: Training predictions from random forest
            y_test_preds_lr: Test predictions from logistic regression
            y_test_preds_rf: Test predictions from random forest
            output_folder: Path to the output folder for saving the images (optional).
    output:
             None
    '''
    # Classification reports
    train_report_lr = classification_report(
        y_train, y_train_preds_lr, output_dict=True)
    test_report_lr = classification_report(
        y_test, y_test_preds_lr, output_dict=True)

    train_report_rf = classification_report(
        y_train, y_train_preds_rf, output_dict=True)
    test_report_rf = classification_report(
        y_test, y_test_preds_rf, output_dict=True)

    # Create classification report plots
    _, ax = plt.subplots(2, 2, figsize=(12, 12))
    sns.heatmap(pd.DataFrame(
        train_report_lr).iloc[:-1, :].T, annot=True, ax=ax[0, 0])
    ax[0, 0].set_title('Logistic Regression Training')
    sns.heatmap(pd.DataFrame(
        test_report_lr).iloc[:-1, :].T, annot=True, ax=ax[0, 1])
    ax[0, 1].set_title('Logistic Regression Testing')
    sns.heatmap(pd.DataFrame(
        train_report_rf).iloc[:-1, :].T, annot=True, ax=ax[1, 0])
    ax[1, 0].set_title('Random Forest Training')
    sns.heatmap(pd.DataFrame(
        test_report_rf).iloc[:-1, :].T, annot=True, ax=ax[1, 1])
    ax[1, 1].set_title('Random Forest Testing')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "classification_reports.png"))
    plt.close()


def feature_importance_plot(model, X_data, output_pth):
    '''
    Creates and stores the feature importances as a plot.

    input:
            model: Trained machine learning model
            X_data: Pandas dataframe containing feature values
            output_pth: Path to output file for saving the plot.

    output:
             None
    '''
    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Create a plot
    plt.figure(figsize=(10, 6))
    plt.bar(range(X_data.shape[1]), importances[indices])
    plt.xticks(range(X_data.shape[1]), [X_data.columns[i]
               for i in indices], rotation=90)
    plt.title('Feature Importances')
    plt.savefig(output_pth)
    plt.close()


def train_models(
        X_train,
        X_test,
        y_train,
        y_test,
        output_model="models",
        output_result="images/results"):
    '''
    Trains logistic regression and random forest models and returns the trained models and predictions.
    input:
              X_train: Training feature data
              X_test: Testing feature data
              y_train: Training target data
              y_test: Testing target data
              output_folder: Path to the output folder for saving models and images (optional).
    output:
              None
    '''
    # Train a logistic regression model
    lr = LogisticRegression(solver='lbfgs', max_iter=3000)
    lr.fit(X_train, y_train)

    # Train a random forest model
    rf = RandomForestClassifier(random_state=42)
    # Perform grid search to find the best hyperparameters
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }
    cv_rfc = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)
    rfc = cv_rfc.best_estimator_

    # Train the best random forest model
    rfc.fit(X_train, y_train)

    # Make predictions
    y_train_preds_lr = lr.predict(X_train)
    y_train_preds_rfc = rfc.predict(X_train)

    y_test_preds_lr = lr.predict(X_test)
    y_test_preds_rfc = rfc.predict(X_test)

    # Save models
    joblib.dump(lr, os.path.join(output_model, "logistic_model.pkl"))
    joblib.dump(rfc, os.path.join(output_model, "random_forest_model.pkl"))

    # ROC curve
    # Plot ROC curve for logistic regression
    RocCurveDisplay.from_estimator(lr, X_test, y_test)
    plt.title('Logistic Regression ROC Curve')
    plt.savefig(os.path.join(output_result, "lr_roc_curve.png"))
    plt.close()

    # Plot ROC curve for random forest
    RocCurveDisplay.from_estimator(rfc, X_test, y_test)
    plt.title('Random Forest ROC Curve')
    plt.savefig(os.path.join(output_result, "rf_roc_curve.png"))
    plt.close()

    # Store classification reports and feature importances
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rfc,
        y_test_preds_lr,
        y_test_preds_rfc,
        output_folder=output_result)

    # Feature importance plot
    feature_importance_plot(
        rfc, X_test, os.path.join(
            output_result, 'feature_importance.png'))

    logger.info('Models trained and saved to models folder')


if __name__ == '__main__':
    # Load data
    logger.info('Loading data')
    DATA = import_data('data/bank_data.csv')

    # Perform EDA and feature engineering
    logger.info('Performing EDA and feature engineering')
    perform_eda(DATA)

    # Perform feature engineering
    logger.info('Performing feature engineering')
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        DATA, 'Churn')

    # Train models
    logger.info('Training models')
    train_models(X_train, X_test, y_train, y_test)

    logger.info('Process complete')
