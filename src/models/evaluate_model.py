from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(y_test, target_pred):

    accuracy = round(accuracy_score(y_test, target_pred)*100, 2)

    class_report = classification_report(y_test, target_pred)

    return accuracy, class_report
