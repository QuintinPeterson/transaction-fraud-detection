from sklearn.preprocessing import StandardScaler

def scale_data(X_train, X_test):
    # Initialize the scaler
    scaler = StandardScaler()

    # Fit the scaler to the feature training data and transform it
    X_train_scaled = scaler.fit_transform(X_train)


    # Transform the feature test data to the standard range
    X_test_scaled = scaler.transform(X_test) 

    return X_train_scaled, X_test_scaled, scaler


