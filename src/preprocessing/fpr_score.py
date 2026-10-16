from sklearn.metrics import confusion_matrix

# false positive rate (FPR)
def fpr_score(y_test, target_pred):
    cm = confusion_matrix(y_test, target_pred)
    tn, fp, _, _ = cm.ravel()
    tnr = tn / (tn + fp)
    return 1 - tnr