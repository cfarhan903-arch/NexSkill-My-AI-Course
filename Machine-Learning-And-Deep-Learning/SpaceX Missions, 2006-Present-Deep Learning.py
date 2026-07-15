import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)
import warnings
warnings.filterwarnings('ignore')

# load the dataset
data = pd.read_csv("database.csv")

# display basic information about the dataset
print("Dataset Shape")
print(data.shape)
print("\n")
print("First Five Rows")
print(data.head())
print("Last Five Rows")
print(data.tail())
print("\n")
print("Columns")
print(data.columns)
print("\n")
print("Data Types")
print(data.dtypes)
print("\n")
print("Dataset Information")
print(data.info())
print("\n")
print("Statistical Summary")

print(data.describe(include='all'))

# print the number of missing values in each column
print("\n")
print("Missing Values")
missing = data.isnull().sum()
print(missing)

plt.figure(figsize=(10,6))
sns.heatmap(
    data.isnull(),
    cbar=False,
    cmap='viridis'
)
plt.title("Missing Values Heatmap")
plt.show()

# print the number of duplicate rows in the dataset
duplicates = data.duplicated().sum()
print("\nDuplicate Rows :", duplicates)

# print the number of unique values in each column
print("\n")
print("Unique Values")
for column in data.columns:
    print("\n")
    print(column)
    print("Unique Values :", data[column].nunique())
    print(data[column].unique())

# print the distribution of the target variable
print("\n")
print("Mission Outcome Distribution")
print(data["Mission Outcome"].value_counts())

plt.figure(figsize=(8,5))
sns.countplot(
    x="Mission Outcome",
    data=data
)
plt.xticks(rotation=45)
plt.title("Mission Outcome Distribution")
plt.show()

# visualize the distribution of the payload mass
plt.figure(figsize=(10,5))
sns.histplot(
    data["Payload Mass (kg)"],
    bins=20,
    kde=True
)
plt.title("Payload Mass Distribution")
plt.show()

# visualize the distribution of the launch year
plt.figure(figsize=(10,5))
sns.countplot(
    y="Vehicle Type",
    data=data,
    order=data["Vehicle Type"].value_counts().index
)
plt.title("Vehicle Type")
plt.show()

# visualize the distribution of the launch site
plt.figure(figsize=(12,6))
sns.countplot(
    y="Launch Site",
    data=data,
    order=data["Launch Site"].value_counts().index
)
plt.title("Launch Site")
plt.show()

# visualize the distribution of the customer countries
plt.figure(figsize=(12,8))
sns.countplot(
    y="Customer Country",
    data=data,
    order=data["Customer Country"].value_counts().index
)
plt.title("Customer Countries")
plt.show()

# visualize the landing outcome
plt.figure(figsize=(8,5))
sns.countplot(
    x="Landing Outcome",
    data=data
)
plt.xticks(rotation=45)
plt.title("Landing Outcome")
plt.show()

# visualiza the payload orbit
plt.figure(figsize=(12,6))
sns.countplot(
    y="Payload Orbit",
    data=data,
    order=data["Payload Orbit"].value_counts().index
)
plt.title("Payload Orbit")
plt.show()

# Correlation Preparation

encoded = data.copy()
encoder = LabelEncoder()
for col in encoded.columns:
    encoded[col] = encoder.fit_transform(
        encoded[col].astype(str)
    )

plt.figure(figsize=(14,10))
sns.heatmap(
    encoded.corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.show()

print("\nEDA Completed Successfully")

# DATA CLEANING & FEATURE ENGINEERING
print("DATA CLEANING")

#  checking the Missing Values
print("\nMissing Values Before Cleaning\n")
print(data.isnull().sum())

# Numerical Columns
numeric_cols = data.select_dtypes(include=['int64','float64']).columns
for col in numeric_cols:
    data[col].fillna(data[col].median(), inplace=True)

# Categorical Columns
categorical_cols = data.select_dtypes(include='object').columns
for col in categorical_cols:
    data[col].fillna(data[col].mode()[0], inplace=True)

print("\nMissing Values After Cleaning\n")
print(data.isnull().sum())

# Remove the Duplicates
before = len(data)
data.drop_duplicates(inplace=True)
after = len(data)
print("\nDuplicates Removed :", before-after)

# Date Feature Engineering
print("\nConverting Launch Date...")
data["Launch Date"] = pd.to_datetime(
    data["Launch Date"],
    errors="coerce"
)
data["Launch Year"] = data["Launch Date"].dt.year
data["Launch Month"] = data["Launch Date"].dt.month
data["Launch Day"] = data["Launch Date"].dt.day
data["Launch Weekday"] = data["Launch Date"].dt.day_name()

# Time Feature Engineering
print("Converting Launch Time...")
data["Launch Time"] = pd.to_datetime(
    data["Launch Time"],
    errors="coerce"
)
data["Launch Hour"] = data["Launch Time"].dt.hour

# Weekend Launch
data["Weekend Launch"] = np.where(
    data["Launch Weekday"].isin(
        ["Saturday","Sunday"]
    ),
    1,
    0
)

# Payload Categories
print("\nCreating Payload Categories")
median_mass = data["Payload Mass (kg)"].median()
data["Heavy Payload"] = np.where(
    data["Payload Mass (kg)"] > median_mass,
    1,
    0
)

# Customer Type Feature
data["International Customer"] = np.where(
    data["Customer Country"]=="USA",
    0,
    1
)

# Falcon Generation
def rocket_generation(vehicle):
    vehicle = str(vehicle)
    if "Falcon 1" in vehicle:
        return 1
    elif "Falcon 9" in vehicle:
        return 2
    elif "Falcon Heavy" in vehicle:
        return 3
    else:
        return 0

data["Rocket Generation"] = data["Vehicle Type"].apply(
    rocket_generation
)

# Display Engineered Features
print("\n")
print(data.head())

# Drop Unnecessary Columns
drop_columns = [
    "Flight Number",
    "Payload Name",
    "Failure Reason",
    "Launch Date",
    "Launch Time"
]

for col in drop_columns:
    if col in data.columns:
        data.drop(col, axis=1, inplace=True)
print("\nRemaining Columns")
print(data.columns)

# Encode All Categorical Columns
from sklearn.preprocessing import LabelEncoder
label_encoders = {}
categorical_columns = data.select_dtypes(include=["object", "string"]).columns
print("Categorical Columns:")
print(categorical_columns)
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(
        data[col].astype(str).fillna("Unknown")
    )
    label_encoders[col] = le
print("\nAfter Encoding:")
print(data.dtypes)
print("\nEncoding Completed")

# Correlation using only numeric columns
numeric_data = data.select_dtypes(include=np.number)
plt.figure(figsize=(15,10))
sns.heatmap(
    numeric_data.corr(),
    cmap="coolwarm",
    annot=True,
    fmt=".2f"
)
plt.title("Correlation Matrix")
plt.show()

# Split Features and Target
X = data.drop("Mission Outcome", axis=1)
y = data["Mission Outcome"]
print("\nFeature Shape")
print(X.shape)
print("\nTarget Shape")
print(y.shape)

# Standard Scaling
standard = StandardScaler()
print(X.dtypes)
X_standard = standard.fit_transform(X)
print("\nStandard Scaling Completed")

# MinMax Scaling
minmax = MinMaxScaler()
X_minmax = minmax.fit_transform(X)
print("MinMax Scaling Completed")

# Train Test Split
X_train_std, X_test_std, y_train, y_test = train_test_split(
    X_standard,
    y,
    test_size=0.20,
    random_state=42
)

X_train_std, X_test_std, y_train, y_test = train_test_split(
    X_standard,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Samples :", len(X_train_std))
print("Testing Samples :", len(X_test_std))
print("\nData Preprocessing Completed Successfully")

# BUILDING MULTIPLE DEEP LEARNING MODELS
print("DEEP LEARNING MODELS")

from tensorflow.keras.utils import to_categorical

# Number of Classes
classes = len(np.unique(y_train))
print("Total Classes :", classes)

# Binary or Multi-Class Classification
if classes > 2:
    y_train_dl = to_categorical(y_train)
    y_test_dl = to_categorical(y_test)
    output_units = classes
    output_activation = "softmax"
    loss_function = "categorical_crossentropy"
else:
    y_train_dl = y_train
    y_test_dl = y_test
    output_units = 1
    output_activation = "sigmoid"
    loss_function = "binary_crossentropy"

# Callbacks
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    verbose=1
)

# MODEL 1
# Baseline ANN
model1 = Sequential()
model1.add(Dense(32,
                 activation="relu",
                 input_shape=(X_train_std.shape[1],)))
model1.add(Dense(output_units,
                 activation=output_activation))
model1.compile(
    optimizer="adam",
    loss=loss_function,
    metrics=["accuracy"]
)
history1 = model1.fit(
    X_train_std,
    y_train_dl,
    validation_split=0.20,
    epochs=50,
    batch_size=8,
    callbacks=[early_stop],
    verbose=1
)

# MODEL 2
# Deep ANN
model2 = Sequential()
model2.add(Dense(64,
                 activation="relu",
                 input_shape=(X_train_std.shape[1],)))
model2.add(Dense(32,
                 activation="relu"))
model2.add(Dense(16,
                 activation="relu"))
model2.add(Dense(output_units,
                 activation=output_activation))
model2.compile(
    optimizer="adam",
    loss=loss_function,
    metrics=["accuracy"]
)
history2 = model2.fit(
    X_train_std,
    y_train_dl,
    validation_split=0.20,
    epochs=75,
    batch_size=8,
    callbacks=[early_stop],
    verbose=1
)

# MODEL 3
# ANN + Dropout
model3 = Sequential()
model3.add(Dense(64,
                 activation="relu",
                 input_shape=(X_train_std.shape[1],)))
model3.add(Dropout(0.30))
model3.add(Dense(32,
                 activation="relu"))
model3.add(Dropout(0.20))
model3.add(Dense(output_units,
                 activation=output_activation))
model3.compile(
    optimizer="adam",
    loss=loss_function,
    metrics=["accuracy"]
)
history3 = model3.fit(
    X_train_std,
    y_train_dl,
    validation_split=0.20,
    epochs=10,
    batch_size=8,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# MODEL 4
# Batch Normalization
model4 = Sequential()
model4.add(Dense(64,
                 activation="relu",
                 input_shape=(X_train_std.shape[1],)))
model4.add(BatchNormalization())
model4.add(Dense(32,
                 activation="relu"))
model4.add(BatchNormalization())
model4.add(Dense(output_units,
                 activation=output_activation))
model4.compile(
    optimizer="adam",
    loss=loss_function,
    metrics=["accuracy"]
)
history4 = model4.fit(
    X_train_std,
    y_train_dl,
    validation_split=0.20,
    epochs=10,
    batch_size=8,
    callbacks=[early_stop],
    verbose=1
)

# MODEL 5
# L2 Regularization
model5 = Sequential()
model5.add(Dense(128,
                 activation="relu",
                 kernel_regularizer=l2(0.001),
                 input_shape=(X_train_std.shape[1],)))
model5.add(Dense(64,
                 activation="relu",
                 kernel_regularizer=l2(0.001)))
model5.add(Dense(32,
                 activation="relu",
                 kernel_regularizer=l2(0.001)))
model5.add(Dense(output_units,
                 activation=output_activation))
model5.compile(
    optimizer="adam",
    loss=loss_function,
    metrics=["accuracy"]
)
history5 = model5.fit(
    X_train_std,
    y_train_dl,
    validation_split=0.20,
    epochs=10,
    batch_size=8,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)
print("\nAll Models Trained Successfully!")

# PART 4
# MODEL EVALUATION & COMPARISON
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# Function to Evaluate Model
results = []
def evaluate_model(model, name):
    print("\n")
    print(name)
    predictions = model.predict(X_test_std)
    # Binary Classification
    if output_units == 1:
        predictions = (predictions > 0.5).astype(int).flatten()
    # Multi-Class Classification
    else:
        predictions = np.argmax(predictions, axis=1)
    if output_units == 1:
        actual = y_test.values
    else:
        actual = np.argmax(y_test_dl, axis=1)
    accuracy = accuracy_score(actual, predictions)
    precision = precision_score(actual, predictions, average="weighted", zero_division=0)
    recall = recall_score(actual, predictions, average="weighted", zero_division=0)
    f1 = f1_score(actual, predictions, average="weighted", zero_division=0)
    
    print("\nAccuracy :", accuracy)
    print("Precision :", precision)
    print("Recall :", recall)
    print("F1 Score :", f1)
    print("\nClassification Report\n")
    print(classification_report(actual, predictions))

    cm = confusion_matrix(actual, predictions)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap="Blues")
    plt.title(name)
    plt.show()

    results.append(
        {
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        }
    )

# Evaluate All Models
evaluate_model(model1,"Baseline ANN")
evaluate_model(model2,"Deep ANN")
evaluate_model(model3,"Dropout ANN")
evaluate_model(model4,"BatchNorm ANN")
evaluate_model(model5,"L2 Regularized ANN")

# Comparison Table
comparison = pd.DataFrame(results)
print("\n")
print("MODEL COMPARISON")
print(comparison)
comparison.sort_values(
    by="Accuracy",
    ascending=False,
    inplace=True
)

plt.figure(figsize=(10,6))
sns.barplot(
    x="Accuracy",
    y="Model",
    data=comparison
)
plt.title("Model Accuracy Comparison")
plt.show()

# Training Curves
def plot_history(history, title):
    plt.figure(figsize=(10,5))
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title(title + " Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["Train","Validation"])
    plt.show()

    plt.figure(figsize=(10,5))
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title(title + " Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Train","Validation"])
    plt.show()

plot_history(history1,"Baseline ANN")
plot_history(history2,"Deep ANN")
plot_history(history3,"Dropout ANN")
plot_history(history4,"BatchNorm ANN")
plot_history(history5,"L2 ANN")

# Select the Best Model
best_model_name = comparison.iloc[0]["Model"]
print("\nBest Model :", best_model_name)
if best_model_name=="Baseline ANN":
    best_model=model1
elif best_model_name=="Deep ANN":
    best_model=model2
elif best_model_name=="Dropout ANN":
    best_model=model3
elif best_model_name=="BatchNorm ANN":
    best_model=model4
else:
    best_model=model5

# Evaluate Best Model on Test Data
loss, accuracy = best_model.evaluate(
    X_test_std,
    y_test_dl,
    verbose=0
)

print("\nBest Model Test Loss:", loss)
print("Best Model Test Accuracy:", accuracy)

# Predictions
prediction = best_model.predict(X_test_std)
if output_units==1:
    prediction=(prediction>0.5).astype(int)
else:
    prediction=np.argmax(prediction,axis=1)
print("\nActual Values")
print(y_test.values[:10])
print("\nPredicted Values")
print(prediction[:10])

# Save the Model
best_model.save("Best_SpaceX_ANN_Model.keras")
print("\nBest model saved successfully!")

# PROJECT SUMMARY
print("SPACE X DEEP LEARNING PROJECT COMPLETED")
print(f"Total Records : {len(data)}")
print(f"Total Features : {X.shape[1]}")
print(f"Target Classes : {classes}")
print(f"Best Model : {best_model_name}")


## Project Summary

'''In this project, I developed a "deep learning classification model" to predict the "Mission Outcome"
using the SpaceX launch dataset. I started by loading the dataset and exploring it by checking its shape, 
data types, missing values, duplicate records, and summary statistics. I also created different visualizations 
to better understand the data, such as the distributions of mission outcomes, vehicle types, launch sites, 
payload mass, customer countries, and payload orbits.'''

'''After understanding the dataset,I cleaned the data by filling missing values, removing duplicate records,
and creating new features such as "Launch Year", "Launch Month", "Launch Hour", "Weekend Launch", 
"Heavy Payload", "International Customer", and "Rocket Generation". I encoded all categorical variables
using "Label Encoding" and applied "StandardScaler" and "MinMaxScaler" to prepare the data for deep learning.'''

'''Next, I trained "five different Artificial Neural Network (ANN) models", including a Baseline ANN, Deep ANN, 
Dropout ANN, Batch Normalization ANN, and L2 Regularized ANN. I used "Early Stopping" and "Reduce Learning Rate" 
callbacks to improve training and reduce overfitting.'''

'''I evaluated all models using "Accuracy, Precision, Recall, F1-Score, Classification Report, and Confusion Matrix".
I compared their performance, selected the best-performing model, and tested it on unseen data. Finally, I saved the best model
for future use and displayed the actual and predicted mission outcomes.'''

'''This project helped me understand the complete deep learning workflow, including data exploration, preprocessing, 
feature engineering,ANN model development, performance evaluation, model comparison, and saving the final trained model.'''

