import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Load the dataset

df = pd.read_csv("yy.csv")

# Identify categorical and numerical columns
categorical_columns = ['Furnishing', 'Locality', 'Status', 'Transaction', 'Type']
numerical_columns = ['Area', 'BHK', 'Bathroom', 'Parking', 'Per_Sqft']

# Define the target variable and features
X = df.drop(columns=['Price'])
y = df['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ])

# Build the RandomForest model pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])

# Define a parameter grid for tuning the RandomForestRegressor
param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [10, 20, 30],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4]
}

# Setup the GridSearchCV with RandomForest and the pipeline
grid_search = GridSearchCV(estimator=model_pipeline, param_grid=param_grid, cv=3, n_jobs=-1, scoring='r2')

# Train the model using grid search
grid_search.fit(X_train, y_train)

# Get the best model from grid search
best_rf_model = grid_search.best_estimator_

# Make predictions with the best model
y_pred_best_rf = best_rf_model.predict(X_test)

# Saving model to disk
pickle.dump(grid_search, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
