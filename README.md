## Solution 

### Preprocessing and Training ( Preprocessing_Training.ipynb )

* Checking the skewness in Dataset
* Finding the Null values in the Dataset
* Dropping the columns which have Null value count above 11000
* Normalizing the data
* Performing Imputation on columns which have Null Values below 11000 using Decision Tree Regressor
* Applying Synthetic Minority Over sampling Technique (SMOTE) to balance the data
* Splitting the Dataset into Train Set and Test Set in 80% and 20% respectively.
* Trained XgBoost classifier with 300 estimators and max_depth 5 (Test Accuracy Score 53.043 %)
* Trained XgBoost classifier with 1000 estimators and max_depth 3 (Test Accuracy Score 53.010 %)
* Trained Random Forest classifier with 400 estimators ( Test Accuracy Score 52.125 % )
* Used the Random Forest classifier for feature selection 
* Used the important feature collected from Random Forest in training another XGBoost classifier with
  200 estimators and max_depth 5 ( Test Accuracy Score 52.65 %)
* Also tried a Support Vector Classifier (Test Accuracy 41.94 %)
* Saving the best model on the disk for future inferences (XgBoost classifier with 300 estimators
  and max_depth 5)  as ( final_model.pickle )

### Testing ( run.py )

* Preprocessing on the test data like dropping the columns which where dropped during training
* Normalizing the test data
* Loading the best XGBoost Regressor model from .pickle file which was saved during training.
* Predicting the final decision associated with an application using this model
* And saving the predicted value as prediction.csv
* Run Command for run.py : < python run.py -d test_data_set >
* test_data_set should be in .csv format
* For more info : < python run.py -h >