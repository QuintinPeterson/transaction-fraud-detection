from sklearn.linear_model import LogisticRegression

def predict_model(model, X_test):
    # predict target values on unknown inputs from the test set
    target_pred = model.predict(X_test)

    return target_pred

def predict_proba(model, X_test):
    prob = model.predict_proba(X_test)

    return prob
