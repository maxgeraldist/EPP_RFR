"""Functions for fitting the regression model."""


def split_data(X, Y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    return X_train, X_test, Y_train, Y_test
def create_random_forest_regressor():
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=100, max_features=2, max_depth=10,
                               min_samples_split=2, min_samples_leaf=1,
                               bootstrap=True, oob_score=False,
                               n_jobs=1, random_state=0,
                               verbose=0)
    return rf
def fit_regressor(rf, X_train, Y_train):
    rf.fit(X_train, Y_train)
    return rf
def evaluate_model(X_train, Y_train, X_test, Y_test):
    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier(n_estimators=100,
                                 bootstrap=True,
                                 oob_score=True)
    rfc.fit(X_train,Y_train)
    test_score = rfc.score(X_test,Y_test)
    print('OOB score:', rfc.oob_score_)
    print('Test score:', test_score)
    return rfc
def predict_and_calculate_mse(rf,X_test,Y_test):
    Y_pred = rf.predict(X_test)
    mse = ((Y_pred - Y_test) ** 2).mean()
    print('MSE before BE = ')
    return mse, Y_pred
def backward_elimination(X,Y):
    import numpy as np
    import statsmodels as sm
    numVars = len(X[0])
    for i in range(0,(numVars-1)):
        regressor_OLS = sm.OLS(Y,X).fit()
        maxVar = max(regressor_OLS.pvalues)
        if maxVar > 0.05:
            for j in range(0,numVars-i):
                if (regressor_OLS.pvalues[j] == maxVar):
                    X = np.delete(X,j,1)
    return X
def crash_not_crash(Y_test, Y_pred): 
    # Print the average predicted value for crashers and non-crashers
    sum_crash=0
    sum_not_crash=0
    count_crash=0
    count_not_crash=0
    for i in range(len(Y_test.values)) :
        if Y_test.values[i] == 0 :
            sum_not_crash+=Y_pred[i]
            count_not_crash+=1
        else:
            sum_crash+=Y_pred[i]
            count_crash+=1
    M=sum_crash/count_crash
    N=sum_not_crash/count_not_crash
    return M,N