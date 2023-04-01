"""Functions plotting results."""

import matplotlib.pyplot as plt

def plot_feature_importances(X, importances, indices, column_titles, name):
    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
            color="r", align="center")
    plt.xticks(range(X.shape[1]), column_titles[indices])
    plt.xlim([-1, X.shape[1]])
    plt.savefig(name)

def plot_predicted_vs_actual(Y_test, Y_pred,name):
    # Plot the predicted values against the actual values
    plt.figure()
    plt.scatter(Y_test, Y_pred)
    plt.xlabel("Actual values")
    plt.ylabel("Predicted values")
    plt.savefig(name)