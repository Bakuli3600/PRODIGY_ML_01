# PRODIGY_ML_01
Implement a linear regression model to predict the prices of houses based on their square footage and the number of bedrooms and bathrooms.    Dataset : - https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

Step 1: Import Libraries
The code imports necessary libraries for data processing, model building, evaluation, and visualization.

'''
import pandas as pd  # For handling datasets (dataframes)
import numpy as np  # For numerical computations
from sklearn.model_selection import train_test_split  # For splitting data into training and validation
from sklearn.linear_model import LinearRegression  # For Linear Regression model
from sklearn.metrics import mean_squared_error  # To evaluate model performance using RMSE
import matplotlib.pyplot as plt  # For visualizing data and predictions
'''

Step 2: Load the Data
The training and test datasets are loaded into pandas dataframes.

'''
train.csv: Contains features and the target variable (SalePrice).
test.csv: Contains only the input features (no SalePrice).
train_df = pd.read_csv('train.csv')  # Training data
test_df = pd.read_csv('test.csv')  # Test data
'''

Step 3: Select Features and Target
You select the features (inputs) and target (output) for training the model.

Features:

GrLivArea: Above-ground living area (square footage).
BedroomAbvGr: Number of bedrooms above ground.
FullBath: Number of full bathrooms.
HalfBath: Number of half bathrooms.
Target:

SalePrice: The price of the house (output variable).

features = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath']
target = 'SalePrice'

Step 4: Handle Missing Values
Missing values in the features are filled with the median of each column.

Filling with median ensures missing values are handled without introducing bias.
Example:
If a feature GrLivArea has missing values [NaN, 2000, 1500, NaN], the median is 1750. Missing values are replaced with 1750.


X = train_df[features].fillna(train_df[features].median())
y = train_df[target]

Step 5: Split the Data into Training and Validation Sets
The training data is split into training and validation sets using an 80-20 split.

Training Set: Used to train the model.
Validation Set: Used to evaluate the model’s performance.

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
test_size=0.2: 20% of the data is used for validation.
random_state=42: Ensures reproducibility of the split.

Step 6: Train the Linear Regression Model
A Linear Regression model is initialized and trained on the training data.

model = LinearRegression()
model.fit(X_train, y_train)
fit() trains the model to find the best-fitting line based on the input features and target.

Step 7: Predict on Validation Set and Calculate RMSE
The model predicts house prices for the validation set.
RMSE (Root Mean Squared Error) is calculated to measure the accuracy of the predictions.

y_pred_val = model.predict(X_val)  # Predictions on validation data
rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
print(f'Validation RMSE: {rmse:.2f}')
RMSE is a measure of how far off the predictions are from the actual values.
Lower RMSE indicates better model performance.

Step 8: Predict on Test Data
The model predicts house prices for the test dataset.

X_test = test_df[features].fillna(test_df[features].median())
test_predictions = model.predict(X_test)
Missing values in the test set are handled in the same way (using median).

Step 9: Save Predictions
The predictions are saved in a CSV file in the required format.

submission = pd.DataFrame({
    'Id': test_df['Id'],  # IDs of the test houses
    'SalePrice': test_predictions  # Predicted sale prices
})
submission.to_csv('submission.csv', index=False)
print("Submission file saved as 'submission.csv'")
The resulting file submission.csv will have two columns:
Id: Identifier for each house.
SalePrice: Predicted price for the house.

Step 10: Visualize Results
The code visualizes the relationship between GrLivArea (square footage) and SalePrice.

Actual Prices are shown as blue dots.
Predicted Prices are shown as red dots.
plt.scatter(X['GrLivArea'], y, color='blue', alpha=0.5, s=5, label='Actual Prices')
plt.scatter(X['GrLivArea'], model.predict(X), color='red', alpha=0.5, s=5, label='Predicted Prices')
plt.xlabel('Above Ground Living Area (Square Feet)')
plt.ylabel('Sale Price')
plt.title('GrLivArea vs SalePrice')
plt.legend()
plt.show()
alpha=0.5: Makes dots semi-transparent.
s=5: Reduces the dot size for clarity.
Summary
This code performs the following:

Loads and preprocesses the data.
Trains a Linear Regression model using selected features.
Validates the model and measures its accuracy using RMSE.
Predicts house prices for the test dataset and saves the results.
Visualizes the relationship between GrLivArea and SalePrice.
Output:
Validation RMSE: Accuracy score for the model.
submission.csv: File containing predictions for test data.
Scatter Plot: Visual representation of actual vs predicted prices.
