import pandas as pd
import joblib
from src.preprocessing.split import split_data
from src.preprocessing.scale import scale_data

# load dataset into a pandas Datafram named 'data'
data = pd.read_csv("data/creditcard_fraud_detection.csv")

# Initialize training and test data from split file
X_train, X_test, y_train, y_test = split_data(data)

# scale feature training data
X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

joblib.dump(scaler, 'scaler.joblib')

#print(X_train_scaled.shape)
#print(X_train_scaled.describe())
print(X_test_scaled.describe())