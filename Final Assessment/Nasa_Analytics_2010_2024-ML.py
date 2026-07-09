import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
import sklearn.svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#XGBoost (not always installed by default, so we guard the import)
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# Load the dataset
data = pd.read_csv('nasa_analytics_2010_2024.csv')

print("First 5 rows of the dataset:")
print(data.head())
print("\nSummary statistics:")
print(data.describe())
print("\nMissing values in each column:")
print(data.isnull().sum())

# Visualize the relationship between 'Launches' and 'Employees'
plt.figure(figsize=(8, 5))
sns.lineplot(x='Launches', y='Employees', data=data, marker='o', color='royalblue')
plt.title('NASA Launches vs Employee Count (2010-2024)')
plt.xlabel('Number of Launches')
plt.ylabel('Number of Employees')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Prepare the data for modeling
columns_to_drop = ['Company', 'Launches', 'Successful', 'Failed', 'Success_Rate_%']
X = data.drop(columns=[col for col in columns_to_drop if col in data.columns])
y = data['Launches']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# i apply 11 models to the dataset, including linear models, tree-based models, and ensemble methods. 
# Each model is wrapped in a pipeline that includes standard scaling where appropriate.

models = {
    'Linear Regression': Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())]),
    'Ridge Regression': Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=1.0, random_state=42))]),
    'Lasso Regression': Pipeline([('scaler', StandardScaler()), ('model', Lasso(alpha=0.1, random_state=42))]),
    'ElasticNet': Pipeline([('scaler', StandardScaler()), ('model', ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42))]),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42, n_estimators=200),
    'Extra Trees': ExtraTreesRegressor(random_state=42, n_estimators=200),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'AdaBoost': AdaBoostRegressor(random_state=42),
    'K-Nearest Neighbors': Pipeline([('scaler', StandardScaler()), ('model', KNeighborsRegressor(n_neighbors=5))]),
    'Support Vector Regressor': Pipeline([('scaler', StandardScaler()), ('model', sklearn.svm.SVR(kernel='rbf', C=10, epsilon=0.1))]),
}

if HAS_XGB:
    models['XGBoost'] = XGBRegressor(random_state=42, n_estimators=200, verbosity=0)
results = []

# i use cv 5-fold cross-validation to evaluate the performance of each model on the training data, 
# and then test them on the hold-out test set.
for name, mdl in models.items():
    mdl.fit(X_train, y_train)
    preds = mdl.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    # Perform 5-fold cross-validation to get a more robust estimation of model performance
    cv_scores = cross_val_score(mdl, X_train, y_train, cv=5, scoring='r2')

    results.append({
        'Model': name,
        'RMSE': rmse,
        'MAE': mae,
        'R2 (test)': r2,
        'R2 (CV mean)': cv_scores.mean(),
        'R2 (CV std)': cv_scores.std()
    })

# Create a DataFrame to summarize the results and sort by cross-validated R² score
results_df = pd.DataFrame(results).sort_values(by='R2 (CV mean)', ascending=False).reset_index(drop=True)

print("\n=== Model Comparison ===")
print(results_df.to_string(index=False))

# Visualize the model comparison based on cross-validated R² scores
plt.figure(figsize=(10, 6))
sns.barplot(data=results_df, x='R2 (CV mean)', y='Model', palette='viridis')
plt.title('Model Comparison: Cross-Validated R² Score')
plt.xlabel('R² Score (5-fold CV mean)')
plt.tight_layout()
plt.show()

# i want to get the best model based on cross-validated R² score and then fine-tune it using GridSearchCV.
best_model_name = results_df.iloc[0]['Model']
print(f"\nBest model based on CV R²: {best_model_name}")
print("Running GridSearchCV to fine-tune Random Forest as the primary candidate...\n")

# Define the parameter grid for Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# i use GridSearchCV to find the best hyperparameters for the Random Forest model, optimizing for R² score.
rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)
print(f"Best Random Forest Params: {grid_search.best_params_}")
print(f"Best Random Forest CV R²: {grid_search.best_score_:.4f}\n")
# Use the best estimator from grid search for predictions and evaluation
model = grid_search.best_estimator_
# Predict on the test set with the optimized model
y_pred = model.predict(X_test)
print(f'Predicted Launches for Test Set: {np.round(y_pred, 2)}')
print(f'Actual Launches for Test Set:    {y_test.values}\n')
# Evaluate the optimized model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Optimized Mean Squared Error: {mse:.4f}')
print(f'Optimized R² Score: {r2:.4f}')

# Feature Importance Visualization
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 5))
sns.barplot(x=importances[indices], y=X.columns[indices], palette="coolwarm")
plt.title("What actually drives 'Launches'? (Feature Importance)")
plt.xlabel("Importance Weight Score")
plt.tight_layout()
plt.show()

# Visualize the actual vs predicted values with a perfect prediction line
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, color='steelblue', s=100, zorder=3, alpha=0.9)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, label='Perfect Prediction Line')
plt.xlabel('Actual Launches')
plt.ylabel('Predicted Launches')
plt.title('Actual vs Predicted Launches')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# predict future NASA launches for 2026 based on the optimized model
print("\n--- Predict Future NASA Launches ---")

future_scenario = pd.DataFrame([{
    'Year': 2026,
    'Revenue_USD_M': 0,              # Placeholder for future revenue
    'Budget_Funding_USD_M': 26000,   # Increased budget estimated for 2026
    'Employees': 19500,              # Higher workforce target
    'Rockets': 5,                    # More rockets in development
}])

future_scenario = future_scenario[X.columns]

future_prediction = model.predict(future_scenario)
print(f"Predicted Launches for 2026 (tuned Random Forest): {future_prediction[0]:.1f} launches")

# Predict using all models for comparison
print("\n--- 2026 Prediction Across All Models ---")
for name, mdl in models.items():
    pred_val = mdl.predict(future_scenario)[0]
    print(f"{name:28s}: {pred_val:.1f} launches")