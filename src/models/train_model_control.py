from sklearn.linear_model import LogisticRegression

def train_model_control(X_train, y_train):
    model_control = LogisticRegression(max_iter=10000)
    model_control.fit(X_train, y_train)

    return model_control