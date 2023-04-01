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
# Add the src directory to the Python path
sys.path.append(str(src_dir))

# Import functions from model.py
from rfr_final.analysis.model import split_data, create_random_forest_regressor, fit_regressor, backward_elimination
from rfr_final.final.plot import plot_feature_importances, plot_predicted_vs_actual

@pytask.mark.depends_on(os.path.join(src_dir, "rfr_final", "data", "data_clean.xlsx"))
@pytask.mark.produces([os.path.join(src_dir.parent, "Results", "feature_importances.jpg"),
                       os.path.join(src_dir.parent, "Results", "predicted_vs_actual.jpg")])
def task_create_plots(depends_on, produces):
    # Load data from CSV file
    df = pd.read_excel(depends_on, sheet_name="Sheet1")
    X= df[['mileage','speed3_100','acc1_100','acc2_100','acc3_100','drg1_100','drg2_100','drg3_100','side1_100','side2_100','side3_100','avg_daily_business_mileage','cor_avg_daily_morning_jam_mileage','cor_avg_daily_night_mileage','avg_speed','max_evening_jam_speed','max_morning_jam_speed','max_night_speed','max_speed']]
    Y= df['crash']
    num_cols = len(X.columns)
    column_titles = X.columns
    X = X.values.reshape(-1, num_cols)
    X = backward_elimination(X,Y)
    # Split the data into training and test sets
    X_train, X_test, Y_train, Y_test = split_data(X, Y)

    # Create a random forest regressor
    rf = create_random_forest_regressor()

    # Fit the model to the training data
    fit_regressor(rf, X_train, Y_train)

    # Get feature importances
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    # Plot the feature importances of the forest
    plot_feature_importances(X, importances, indices, column_titles, produces[0])

    # Predict on the test set
    Y_pred = rf.predict(X_test,)

    # Plot the predicted values against the actual values
    plot_predicted_vs_actual(Y_test, Y_pred, produces[1])