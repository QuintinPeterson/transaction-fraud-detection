from sklearn.preprocessing import StandardScaler

def scale_data(X_train, X_test):
    # Initialize the scaler
    scaler = StandardScaler()

    # Fit the scaler to the data and transform it
    X_train = scaler.fit_transform(X_train)


    # Transform the test data to the standard range
    X_test = scaler.transform(X_test)

    return X_train, X_test


