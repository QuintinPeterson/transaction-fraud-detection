import pandas as pd
from src.preprocessing.split import split_data
from src.preprocessing.scale import scale_data

# load dataset into a pandas Datafram named 'data'
data = pd.read_csv("data/creditcard_fraud_detection.csv")

# Initialize training and test data from split file
X_train, X_test, y_train, y_test = split_data(data)

X_train, X_test = scale_data(X_train, X_test)
