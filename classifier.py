import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score
import _pickle as Pickle
import sys


def preprocessing(data):
    # drop unused columns
    clean_data = data.drop(columns=['variable17', 'variable18',  "variable1"])
    # remove NAs
    clean_data = clean_data.dropna(axis=0)
    # convert categorical variables into dummy variables
    clean_data = pd.get_dummies(clean_data, drop_first=True)
    # ensure that the test set columns match the training set columns
    f = open("columns.txt", "r")
    columns = f.read().splitlines()
    missing_columns = set(columns) - set(clean_data.columns)
    for c in missing_columns:
        clean_data[c] = 0

    clean_data = clean_data[columns]

    return clean_data


path = ""
if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    path = 'binary_classifier_data\\validation.csv'


validation_data_raw = pd.read_csv(path, sep=';', header=0, decimal=",")
validation_data = preprocessing(validation_data_raw)
X_test = validation_data.iloc[:, :-1]
y_test = validation_data.iloc[:, -1]

# load the classification model
RF = Pickle.load(open("random_forest.pkl", 'rb'))
y_predicted = RF.predict(X_test)

tn, fp, fn, tp = confusion_matrix(y_test, y_predicted).ravel()
print("tp:", tp, "fp", fp)
print("fn:", fn, "tn", tn)
precision = precision_score(y_test, y_predicted)
recall = recall_score(y_test, y_predicted)
accuracy = accuracy_score(y_test, y_predicted)
print("Precision", precision*100)
print("Recall", recall*100)
print("Accuracy", accuracy*100)
