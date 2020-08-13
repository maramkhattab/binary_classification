#Getting Started
# Installation
Install the dependencies.
```sh
pip install -r requirements.txt
```
##Building the model
A random forest model is used for binary classifcation. The model is trained on the training set in "binary_classifier_data" and then saved to "random_forest.pkl". 
To run the script run the following command

```sh
python building_the_model.py
```
## Running the tests 
To test the classifier using the validation set in "binary_classifier_data" folder  
```sh
python classifier.py 
```
To test the classifier using a different test set
```sh
python classifier.py  FILE_PATH
```