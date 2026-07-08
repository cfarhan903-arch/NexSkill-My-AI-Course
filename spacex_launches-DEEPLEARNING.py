import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# load the dataset
data = pd.read_csv("spacex_launches.csv")

# explore the dataset
print(data.head())
print("\nDataset Shape:")
print(data.shape)
print("\nInformation")
print(data.info())
print("\nSummary Statistics")
print(data.describe(include='all'))
print("\nMissing Values")
print(data.isnull().sum())
data.drop_duplicates(inplace=True)

# handle missing values
for col in data.columns:
    if data[col].dtype == "object":
        data[col] = data[col].fillna(data[col].mode()[0])
    else:
        data[col] = pd.to_numeric(data[col], errors="coerce")
        data[col] = data[col].fillna(data[col].median())

# feature engineering
data["date_utc"] = pd.to_datetime(data["date_utc"])
data["launch_year"] = data["date_utc"].dt.year
data["launch_month"] = data["date_utc"].dt.month
data.drop("date_utc", axis=1, inplace=True)

# encode categorical variables
encoder = LabelEncoder()
for col in data.select_dtypes(include="object").columns:
    data[col] = encoder.fit_transform(data[col].astype(str))

# feature selection
X = data.drop("success", axis=1)
y = data["success"]

# scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# build the deep learning model
model = Sequential()
model.add(Dense(64, activation="relu", input_shape=(X_train.shape[1],)))
model.add(Dropout(0.30))
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.20))
model.add(Dense(16, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

# compile the model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# model summary
model.summary()

# set early stopping
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

# train the model
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)

# evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)

print("\nTest Loss :", loss)
print("Test Accuracy :", accuracy)

# predict on the test set
probabilities = model.predict(X_test)
predictions = (probabilities > 0.5).astype(int)

# check accuracy score
acc = accuracy_score(y_test, predictions)
print("\nAccuracy Score")
print(acc)

# plot confusion matrix
cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# print classification report
print(classification_report(y_test, predictions))

# plot training and validation accuracy
plt.figure(figsize=(8,5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.show()

# plot training and validation loss
plt.figure(figsize=(8,5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.show()

# roc curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, probabilities)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(7,6))
plt.plot(fpr, tpr, label="AUC = %0.2f" % roc_auc)
plt.plot([0,1],[0,1],'r--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# save the model
model.save("spacex_model.keras")
print("\nModel Saved Successfully!")

# predict a single sample
sample = X_test[:1]
prediction = model.predict(sample)
print("\nPrediction Probability :", prediction)

if prediction > 0.5:
    print("Predicted Launch: SUCCESS")
else:
    print("Predicted Launch: FAILURE")