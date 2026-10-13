import pandas as pd
from sklearn.preprocessing import StandardScaler

def scale_data(X_train, X_test):
    # Initialize the scaler
    scaler = StandardScaler()

    # Fit the scaler to the feature training data and transform it
    X_train_scaled = scaler.fit_transform(X_train)

    # fit_transform turned X_trained_scaled into an array so we turn it back into a df
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)


    # Transform the feature test data to the standard range
    X_test_scaled = scaler.transform(X_test) 

    # same as X_trained_scaled, turn it back into a df from an array
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    return X_train_scaled, X_test_scaled, scaler


