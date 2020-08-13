import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2


def preprocessing(data):

    #clean_data = data.drop(columns=['variable17', 'variable18', "variable3", "variable1", "variable6"])
    clean_data = data.drop(columns=['variable17', 'variable18'])
    clean_data = clean_data.dropna(axis=0)
    clean_data = pd.get_dummies(clean_data, drop_first=True)
    return clean_data


training_data_raw = pd.read_csv(
    'binary_classifier_data\\training.csv', sep=';', header=0, decimal=",")

training_data = preprocessing(training_data_raw)
validation_data_raw = pd.read_csv(
    'binary_classifier_data\\validation.csv', sep=';', header=0, decimal=",")
validation_data = preprocessing(validation_data_raw)

# matching columns in train and validation sets
missing_columns = set(training_data.columns) - set(validation_data.columns)
for c in missing_columns:
    validation_data[c] = 0
validation_data = validation_data[training_data.columns]

X_train = training_data.iloc[:, :-1]
Y_train = training_data.iloc[:, -1]
X_test = validation_data.iloc[:, :-1]
Y_test = validation_data.iloc[:, -1]

# LR = LogisticRegression(random_state=0, solver='lbfgs',
#                         multi_class='ovr').fit(X_train, Y_train)

# Y_predicted = LR.predict(X_test)
# tn, fp, fn, tp = confusion_matrix(Y_test, Y_predicted).ravel()
# print("tp:", tp, "fp", fp)
# print("fn:", fn, "tn", tn)

# print("SVM")

# SVM = svm.LinearSVC()
# SVM.fit(X_train, Y_train)
# Y_predicted = SVM.predict(X_test)

# tn, fp, fn, tp = confusion_matrix(Y_test, Y_predicted).ravel()
# print("tp:", tp, "fp", fp)
# print("fn:", fn, "tn", tn)

print("Random forest")

RF = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
RF.fit(X_train, Y_train)
Y_predicted = RF.predict(X_test)

tn, fp, fn, tp = confusion_matrix(Y_test, Y_predicted).ravel()
print("tp:", tp, "fp", fp)
print("fn:", fn, "tn", tn)

# fs = SelectKBest(score_func=chi2, k='all')
# fs.fit(X_train, Y_train)
# for i in range(len(fs.scores_)):
#     print(X_train.columns[i], ":", fs.scores_[i])

# print(training_data["variable2"])
# print(validation_data.head())
