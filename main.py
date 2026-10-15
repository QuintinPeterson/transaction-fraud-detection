import pandas as pd
import joblib
from src.preprocessing.split import split_data
from src.preprocessing.scale import scale_data
from src.preprocessing.resample import resample_data
from src.models.train_model_control import train_model_control
from src.models.train_model import train_model
from src.models.predict_model import predict_model, predict_proba
from src.models.evaluate_model import evaluate_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# load dataset into a pandas Datafram named 'data'
data = pd.read_csv("data/creditcard_fraud_detection.csv")

# Initialize training and test data from split file
X_train, X_test, y_train, y_test = split_data(data)

# scale feature training data
X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

# save scaler for scaling future real-time data
joblib.dump(scaler, 'scaler.joblib')

# resample all training data
X_train_resampled, y_train_resampled = resample_data(X_train_scaled, y_train)

# train control model for comparison
#model_control = train_model_control(X_train, y_train)

# predict control model
#target_control_pred = predict_model(model_control, X_test)
# accuracy of model on the testing set
#accuracy_control, class_report_control =  evaluate_model(y_test, target_control_pred)
#print(f"Accuracy BEFORE SMOTE: {accuracy_control}" "%\n")

# classification report to measure how well the model is performing
#print("Classification Report BEFORE SMOTE:\n", class_report_control)

# train model with SMOTE
model = train_model(X_train_resampled, y_train_resampled)

# predict model
target_pred = predict_model(model, X_test_scaled)

# accuracy of model on the testing set
accuracy, class_report =  evaluate_model(y_test, target_pred)

'''
print(f"Accuracy AFTER SMOTE: {accuracy}" "%\n")

# classification report to measure how well the model is performing
print("Classification Report AFTER SMOTE:\n", class_report)

sns.heatmap(confusion_matrix(y_test, target_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix AFTER SMOTE")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
'''
prob = predict_proba(model, X_test_scaled)
print(prob[:5])      # first 5 rows: you should see decimals in two columns
print(prob.shape)