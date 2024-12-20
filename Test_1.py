# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Step 1: Load the training and test datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Step 2: Select relevant features and target
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath']  # Input features
target = 'SalePrice'  # Target variable

# Step 3: Handle missing values in the training set
X = train_df[features].fillna(train_df[features].median())  # Fill missing values with median
y = train_df[target]

# Step 4: Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Predict on the validation set and calculate RMSE
y_pred_val = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
print(f'Validation RMSE: {rmse:.2f}')

# Step 7: Train the model on the full training set and predict on test data
X_test = test_df[features].fillna(test_df[features].median())  # Handle missing values in test set
test_predictions = model.predict(X_test)

# Step 8: Save the predictions to a submission file
submission = pd.DataFrame({
    'Id': test_df['Id'],
    'SalePrice': test_predictions
})
submission.to_csv('submission.csv', index=False)
print("Submission file saved as 'submission.csv'")

# Step 9: Visualize the relationship between GrLivArea and SalePrice
plt.scatter(X['GrLivArea'], y, color='blue', alpha=0.5, s=5, label='Actual Prices')  # Set size with 's'
plt.scatter(X['GrLivArea'], model.predict(X), color='red', alpha=0.5, s=5, label='Predicted Prices')
plt.xlabel('Above Ground Living Area (Square Feet)')
plt.ylabel('Sale Price')
plt.title('GrLivArea vs SalePrice')
plt.legend()
plt.show()

