import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

# load the dataset
df = pd.read_csv('spacex.csv')

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nDataset Description:")
print(df.describe())
print("\nMissing Values in Each Column:")
print(df.isnull().sum())

# Convert 'Datetime' column to datetime type and sort the DataFrame
df['Datetime'] = pd.to_datetime(df['Datetime'])
df = df.sort_values(by='Datetime').reset_index(drop=True)

# Feature Engineering
df['Lag_1'] = df['Close'].shift(1)
df['Lag_2'] = df['Close'].shift(2)
df['Lag_3'] = df['Close'].shift(3)

# Moving Averages
df['MA_5'] = df['Close'].rolling(5).mean()
df['MA_10'] = df['Close'].rolling(10).mean()

# Exponential Moving Average
df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()

# Volatility
df['Vol_5'] = df['Close'].rolling(5).std()

# Daily Returns
df['Return'] = df['Close'].pct_change()

# RSI (14)

delta = df['Close'].diff()

gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()

rs = gain / loss

df['RSI'] = 100 - (100 / (1 + rs))

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# Correlation Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

# Split the dataset into training and testing sets (80% train, 20% test)
split_idx = int(len(df) * 0.8)

features = [
    'Lag_1',
    'Lag_2',
    'Lag_3',
    'MA_5',
    'MA_10',
    'EMA_5',
    'EMA_10',
    'Vol_5',
    'Return',
    'RSI',
    'Volume'
]
target = 'Close'

X_train = df.iloc[:split_idx][features]
y_train = df.iloc[:split_idx][target]

X_test = df.iloc[split_idx:][features]
y_test = df.iloc[split_idx:][target]

tscv = TimeSeriesSplit(n_splits=5)

rf_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5]
}

rf_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    rf_grid,
    cv=tscv,
    scoring='r2',
    n_jobs=-1
)

rf_search.fit(X_train, y_train)

print("Best Random Forest Parameters:")
print(rf_search.best_params_)

# Define a dictionary of models to evaluate
models = {
    'Linear Regression': Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())]),
    'Ridge Regression': Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=1.0))]),
    'Lasso Regression': Pipeline([('scaler', StandardScaler()), ('model', Lasso(alpha=0.1))]),
    'ElasticNet': Pipeline([('scaler', StandardScaler()), ('model', ElasticNet(alpha=0.1, l1_ratio=0.5))]),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': rf_search.best_estimator_,
    'Extra Trees': ExtraTreesRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'AdaBoost': AdaBoostRegressor(random_state=42),
    'K-Nearest Neighbors': Pipeline([('scaler', StandardScaler()), ('model', KNeighborsRegressor(n_neighbors=5))]),
    'Support Vector Regressor': Pipeline([('scaler', StandardScaler()), ('model', SVR(kernel='rbf', C=10, epsilon=0.1))])
}

# Evaluate each model and store the results
results = []
for name, model in models.items():

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    mape = mean_absolute_percentage_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"\n{name}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"MAPE : {mape:.4f}")
    print(f"R2   : {r2:.4f}")
    print("-"*40)

    results.append({
        'Model': name,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2
    })

# Create a DataFrame to summarize the results and sort by R2 score
results_df = pd.DataFrame(results).sort_values(by='R2', ascending=False).reset_index(drop=True)
print("\n=== Model Evaluation Summary ===")
print(results_df.to_string(index=False))
results_df.to_csv("Model_Results.csv", index=False)
print("Results saved to Model_Results.csv")

# check the best model based on R2 score
best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]
best_preds = best_model.predict(X_test)

best_row = results_df.iloc[0]

print("\n===== Best Model Summary =====")
print(f"Model : {best_model_name}")
print(f"RMSE  : {best_row['RMSE']:.4f}")
print(f"MAE   : {best_row['MAE']:.4f}")
print(f"MAPE  : {best_row['MAPE']:.4f}")
print(f"R2    : {best_row['R2']:.4f}")

# Feature Importance
if hasattr(best_model, "feature_importances_"):

    importance = pd.Series(
        best_model.feature_importances_,
        index=features
    ).sort_values()

    plt.figure(figsize=(8,6))

    importance.plot(kind="barh")

    plt.title("Feature Importance")

    plt.xlabel("Importance Score")

    plt.show()

# Save the best model to a file using joblib    
joblib.dump(best_model, "best_spacex_model.pkl")
print("Model Saved Successfully")

# draw a bar plot to visualize the R2 scores of all models
plt.figure(figsize=(12, 6))
sns.barplot(x='R2', y='Model', data=results_df, palette='viridis')
plt.title('Model R2 Score Comparison')
plt.xlabel('R2 Score')
plt.ylabel('Model')
plt.xlim(0, 1)
plt.grid(axis='x', alpha=0.3)
# Annotate the bars with R2 values
for index, row in results_df.iterrows():
    plt.text(row['R2'] + 0.01, index, f"{row['R2']:.4f}", color='black', va='center')
plt.tight_layout()
plt.show()

# Visualize the actual vs predicted prices for the best model
plt.figure(figsize=(12, 6))
plt.plot(df.loc[split_idx:, 'Datetime'], y_test.values, label='Actual Price', color='royalblue')
plt.plot(df.loc[split_idx:, 'Datetime'], best_preds, label=f'Predicted Price ({best_model_name})', color='crimson', linestyle='--')
plt.title('SpaceX (SPCX) Stock Forecast Model Evaluation')
plt.xlabel('Datetime')
plt.ylabel('Price')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Residual Plot
residuals = y_test - best_preds
plt.figure(figsize=(10,5))
plt.scatter(best_preds, residuals)
plt.axhline(0, color='red')
plt.xlabel("Predicted Price")
plt.ylabel("Residual Error")
plt.title("Residual Plot")
plt.grid(True)
plt.show()

# Create a simulated future input data point for the next 30-minute block
future_inputs = pd.DataFrame([{
    'Lag_1': df.iloc[-1]['Close'],
    'Lag_2': df.iloc[-2]['Close'],
    'Lag_3': df.iloc[-3]['Close'],
    'MA_5': df.iloc[-5:]['Close'].mean(),
    'MA_10': df.iloc[-10:]['Close'].mean(),
    'EMA_5': df.iloc[-1]['EMA_5'],
    'EMA_10': df.iloc[-1]['EMA_10'],
    'Vol_5': df.iloc[-5:]['Close'].std(),
    'Return': df.iloc[-1]['Return'],
    'RSI': df.iloc[-1]['RSI'],
    'Volume': df.iloc[-5:]['Volume'].mean()
}])                

# Ensure the columns follow the exact training layout order
future_inputs = future_inputs[features]

# Run future predictions across all trained models
print("\n--- Next 30-Minute Price Predictions ---")
for name, model in models.items():
    future_pred = model.predict(future_inputs)[0]
    print(f"{name:28s}: ${future_pred:.2f}")

# Isolate the top-performing model's explicit prediction
champion_pred = best_model.predict(future_inputs)[0]
future_result = pd.DataFrame({
    "Prediction Time": [df.iloc[-1]["Datetime"]],
    "Predicted Close": [champion_pred]
})
future_result.to_csv("Future_Prediction.csv", index=False)
print("Future prediction saved to Future_Prediction.csv")