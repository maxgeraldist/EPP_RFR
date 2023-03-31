import pandas as pd
import pytask
import os
import sys
current_file_dir = os.path.abspath(os.path.dirname(__file__))
src_dir = os.path.join(current_file_dir)
sys.path.append(src_dir)
# Import functions from model.py
from model import split_data, create_random_forest_regressor, fit_regressor, evaluate_model, predict_and_calculate_mse, backward_elimination, crash_not_crash
import pytask
@pytask.mark.depends_on(os.path.join(src_dir, "..", "data_management", "data_clean.xlsx"))
@pytask.mark.produces("results.txt")
def task_fit_evaluate_model(depends_on, produces):
    # Load data from CSV file
    data = pd.read_xlsx(depends_on, sheet_name="JTI_weekly_prepared_26_11_2017")
    X = data.drop('target', axis=1)
    Y = data['target']
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
        f.write('Model before BE\n')
        f.write(f'OOB score: {rfc.oob_score_}\n')
        f.write(f'Test score: {rfc.score(X_test,Y_test)}\n')
        f.write(f'MSE: {mse}\n')
        f.write(f'Average predicted value for crashers: {M}\n')
        f.write(f'Average predicted value for non-crashers: {N}\n')
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
        mse,_ = predict_and_calculate_mse(rf,X_test,Y_test)
        f.write('Model after BE\n')
        f.write(f'OOB score: {rfc.oob_score_}\n')
        f.write(f'Test score: {rfc.score(X_test,Y_test)}\n')
        f.write(f'MSE: {mse}\n')
        f.write(f'Average predicted value for crashers: {M}\n')
        f.write(f'Average predicted value for non-crashers: {N}\n')