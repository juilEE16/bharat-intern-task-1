# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load your dataset
# Assuming you have a CSV file named 'house_prices.csv' with features and target variable
data = pd.read_csv('house_prices.csv')

# Split the data into features and target variable
X = data.drop('target_variable_name', axis=1)  # Replace 'target_variable_name' with your actual target variable
y = data['target_variable_name']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)
linear_predictions = linear_model.predict(X_test_scaled)

# Random Forest model
rf_model = RandomForestRegressor()
rf_model.fit(X_train_scaled, y_train)
rf_predictions = rf_model.predict(X_test_scaled)

# Neural Network model using TensorFlow/Keras
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_data=(X_test_scaled, y_test))

# Evaluate the models
linear_rmse = np.sqrt(mean_squared_error(y_test, linear_predictions))
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
nn_predictions = model.predict(X_test_scaled)
nn_rmse = np.sqrt(mean_squared_error(y_test, nn_predictions))

print(f'Linear Regression RMSE: {linear_rmse}')
print(f'Random Forest RMSE: {rf_rmse}')
print(f'Neural Network RMSE: {nn_rmse}')
