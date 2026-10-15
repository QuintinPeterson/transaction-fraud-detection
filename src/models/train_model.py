from sklearn.linear_model import LogisticRegression

def train_model(X_train_resampled, y_train_resampled):
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train_resampled, y_train_resampled)

    return model