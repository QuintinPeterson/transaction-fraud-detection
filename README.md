TRANSACTION FRAUD DETECTION SYSTEM

This project is a series of Machine Learning models trained on real open-source credit card data to spot and flag unseen fraudulent purchases. It uses an array of different models and techniques to tune and check efficiency.

Setup: Download the dataset from Kaggle - https://www.kaggle.com/datasets/gzdekzlkaya/credit-card-fraud-detection-dataset

Install required libraries run: pip install -r requirements.txt

A control Model trained on a raw dataset under the cost assumptions that a missed fraud transaction costs $10,000 and a falsely flagged fraud transaction cost $50 for a bank, would have lost the bank $340,650. While the model trained on SMOTE resampled data would only lose less than half, around $153,350, even with a slightly lower accuracy rating.

Control Model: Accuracy BEFORE SMOTE: 99.92%

Classification Report BEFORE SMOTE:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00     56864
           1       0.83      0.65      0.73        98

SMOTE Model (UNTUNED CUTOFF): Accuracy AFTER SMOTE: 97.41%

Classification Report AFTER SMOTE:
               precision    recall  f1-score   support

           0       1.00      0.97      0.99     56864
           1       0.06      0.92      0.11        98 