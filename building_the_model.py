
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score
import _pickle as Pickle

training_data = pd.read_csv(
    'binary_classifier_data\\training.csv', sep=';', header=0, decimal=",")
# observing the first 5 rows of the data
training_data.head()
# it can be observed that variable 17 is a multiple of variable 4 so it should be dropped
training_data = training_data.drop(columns=["variable17"])
# counting the number of NAs in each column
print(len(training_data) - training_data.count())
# variable 18 had 2145 NA values which is a large percentage from the # of recrods
training_data = training_data.drop(columns=["variable18"])
# Other than variable 18, the number of NAs in each columns is not large which emans we can drop rows that have missing values without affecting the model's performance
training_data = training_data.dropna(axis=0)
# transforming categorical variables into dummy variables
training_data_dummy = pd.get_dummies(training_data, drop_first=True)

# Training a random forrest model
X_train = training_data_dummy.iloc[:, :-1]
y_train = training_data_dummy.iloc[:, -1]
RF = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
RF.fit(X_train, y_train)
# checking features importance
importance = RF.feature_importances_

for i, v in enumerate(importance):
    print(X_train.columns[i], "Score: %.5f" % (v))

# variable 1 had 0 importance therefore it should be removed from the model

training_data = training_data.drop(columns=["variable1"])

# retrain the model using the new variables

training_data_dummy = pd.get_dummies(training_data, drop_first=True)
X_train = training_data_dummy.iloc[:, :-1]
y_train = training_data_dummy.iloc[:, -1]
RF = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
RF.fit(X_train, y_train)

# save the model
Pickle.dump(RF, open("random_forest.pkl", 'wb'))
# save the columns of the training set to use it to match the columns of the validation set
with open('columns.txt', 'w') as f:
    # for column in training_data_dummy.columns:
    f.write('\n'.join(training_data_dummy.columns))
