import pandas as pd
import pytask
import os
import sys
from pathlib import Path
# Get the absolute path of the parent directory of the current file
current_file_dir = Path(__file__).resolve().parent
# Construct the absolute path of the src directory
src_dir = current_file_dir.parent.parent
sys.path.append(str(src_dir))
# Import functions from model.py
from rfr_final.analysis.model import split_data, create_random_forest_regressor, fit_regressor, evaluate_model, predict_and_calculate_mse, backward_elimination, crash_not_crash
import pytask
sys.path.append(str(src_dir))
# Import functions from model.py
from rfr_final.analysis.model import split_data, create_random_forest_regressor, fit_regressor, backward_elimination
from rfr_final.final.plot import plot_feature_importances, plot_predicted_vs_actual

@pytask.mark.depends_on(os.path.join(src_dir, "rfr_final", "data", "data_clean.xlsx"))
@pytask.mark.produces(os.path.join(src_dir.parent, "Results","Results.txt" ))
def task_fit_evaluate_model(depends_on, produces):
    # Load data from xlsx file
    df = pd.read_excel(depends_on, sheet_name="Sheet1")
    X= df[['mileage','speed3_100','acc1_100','acc2_100','acc3_100','drg1_100','drg2_100','drg3_100','side1_100','side2_100','side3_100','avg_daily_business_mileage','cor_avg_daily_morning_jam_mileage','cor_avg_daily_night_mileage','avg_speed','max_evening_jam_speed','max_morning_jam_speed','max_night_speed','max_speed']]
    Y= df['crash']
    num_cols = len(X.columns)
    X = X.values.reshape(-1, num_cols)
    # Split the data into training and test sets
    X_train, X_test, Y_train, Y_test = split_data(X, Y)
    # Create a random forest regressor
    rf = create_random_forest_regressor()
    # Fit the model to the training data
    fit_regressor(rf, X_train, Y_train)
    # Evaluate the model on the test data
    rfc = evaluate_model(X_train,Y_train,X_test,Y_test)
    # Predict on the test set and calculate the mean squared error
    mse, Y_pred = predict_and_calculate_mse(rf,X_test,Y_test)
    # Calculate average predicted value for crashers and non-crashers
    M,N = crash_not_crash(Y_test,Y_pred)
    # Write results to file
    
    with open(produces, 'w') as f:
        column_titles = df.columns.values
        importances = rf.feature_importances_
        import numpy as np
        indices = np.argsort(importances)[::-1]
        f.write('Feature Importances:\n')
        for i in range(X.shape[1]):
            f.write("%d. feature %s (%f)\n" % (i + 1, column_titles[indices[i]], importances[indices[i]]))
        f.write('Model before BE\n')
        f.write(f'OOB score: {rfc.oob_score_}\n')
        f.write(f'Test score: {rfc.score(X_test,Y_test)}\n')
        f.write(f'MSE: {mse}\n')
        f.write(f'Average predicted value for crashers: {M}\n')
        f.write(f'Average predicted value for non-crashers: {N}\n')
        # Write feature importances to file
        import numpy as np
        # Perform backward elimination on data
        X = backward_elimination(X,Y)
        # Split the data into training and test sets
        X_train, X_test, Y_train, Y_test = split_data(X,Y)
        # Create a random forest regressor
        rf = create_random_forest_regressor()
        # Fit the model to the training data
        fit_regressor(rf,X_train,Y_train)
        # Evaluate the model on the test data
        rfc = evaluate_model(X_train,Y_train,X_test,Y_test)
        # Predict on the test set and calculate the mean squared error
        mse,Y_pred = predict_and_calculate_mse(rf,X_test,Y_test)
        M,N = crash_not_crash(Y_test,Y_pred)
        # Write results to file
        column_titles = df.columns.values
        importances = rf.feature_importances_
        import numpy as np
        indices = np.argsort(importances)[::-1]
        f.write('Feature Importances:\n')
        for i in range(X.shape[1]):
            f.write("%d. feature %s (%f)\n" % (i + 1, column_titles[indices[i]], importances[indices[i]]))
        f.write('Model after BE\n')
        f.write(f'OOB score: {rfc.oob_score_}\n')
        f.write(f'Test score: {rfc.score(X_test,Y_test)}\n')
        f.write(f'MSE: {mse}\n')
        f.write(f'Average predicted value for crashers: {M}\n')
        f.write(f'Average predicted value for non-crashers: {N}\n')