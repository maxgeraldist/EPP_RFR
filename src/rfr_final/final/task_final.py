"""Tasks running the results formatting (tables, figures)."""

import pandas as pd
import pytask
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import numpy as np
import os
# Get the absolute path of the parent directory of the current file
current_file_dir = Path(__file__).resolve().parent

# Construct the absolute path of the src directory
src_dir = current_file_dir.parent.parent
print (src_dir)
# Add the src directory to the Python path
sys.path.append(str(src_dir))

# Import functions from model.py
from rfr_final.analysis.model import split_data, create_random_forest_regressor, fit_regressor, evaluate_model, predict_and_calculate_mse

@pytask.mark.depends_on(os.path.join(src_dir, "..", "data_management", "data_clean.xlsx"))
@pytask.mark.produces(["feature_importances.jpg", "predicted_vs_actual.jpg"])
def task_create_plots(depends_on, produces):
    # Load data from CSV file
    data = pd.read_excel(depends_on, sheet_name="JTI_weekly_prepared_26_11_2017")
    X = data.drop('target', axis=1)
    Y = data['target']

    # Split the data into training and test sets
    X_train, X_test, Y_train, Y_test = split_data(X, Y)

    # Create a random forest regressor
    rf = create_random_forest_regressor()

    # Fit the model to the training data
    fit_regressor(rf, X_train, Y_train)

    # Get feature importances
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    column_titles = X.columns

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
            color="r", align="center")
    plt.xticks(range(X.shape[1]), column_titles[indices])
    plt.xlim([-1, X.shape[1]])
    plt.savefig(produces[0])

    # Predict on the test set
    Y_pred = rf.predict(X_test)

    # Plot the predicted values against the actual values
    plt.figure()
    plt.scatter(Y_test, Y_pred)
    plt.xlabel("Actual values")
    plt.ylabel("Predicted values")
    plt.savefig(produces[1])