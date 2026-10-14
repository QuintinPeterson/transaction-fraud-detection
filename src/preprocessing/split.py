from sklearn.model_selection import train_test_split

def split_data(data):
    # seperate features (x) from target value (y)
    X = data.drop("Class", axis=1)
    y = data["Class"]

    # create test and training subsets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test