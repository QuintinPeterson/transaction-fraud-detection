from imblearn.over_sampling import SMOTE

def resample_data(X_train_scaled, y_train):

    # create smote sampling method
    smote = SMOTE(sampling_strategy='minority', random_state=42)

    # resample the training data on the features and targets
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    return X_train_resampled, y_train_resampled
