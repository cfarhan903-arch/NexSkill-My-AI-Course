import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import metrics

# load the dataset
data = pd.read_csv("heart.csv")

print(data.head())
print(data.info())

sns.set_style("whitegrid")

plt.figure(figsize=(10,6))
sns.lineplot(x="MaxHR", y="Cholesterol", data=data)
plt.title("MaxHR vs Cholesterol")
plt.tight_layout()
plt.show()

#Encode Categorical Columns
le = LabelEncoder()
categorical_cols = [
    "Sex",
    "ChestPainType",
    "RestingECG",
    "ExerciseAngina",
    "ST_Slope"
]

for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

print("\nEncoded Dataset")
print(data.head())
data.to_csv("clean_heart.csv", index=False)

# Features and Target
X = data.drop("HeartDisease", axis=1)
y = data["HeartDisease"]
print("\nFeatures")
print(X.head())
print("\nTarget")
print(y.head())

# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=16,
    stratify=y
)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression
logreg = LogisticRegression(
    random_state=16,
    max_iter=1000
)
logreg.fit(X_train, Y_train)
log_pred = logreg.predict(X_test)
print("Logistic Regression")
print("Accuracy:",
      metrics.accuracy_score(Y_test, log_pred))
print(metrics.classification_report(Y_test, log_pred))

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, Y_train)
dt_pred = dt.predict(X_test)
print("Decision Tree")
print("Accuracy:",
      metrics.accuracy_score(Y_test, dt_pred))
print(metrics.classification_report(Y_test, dt_pred))

# Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf.fit(X_train, Y_train)
rf_pred = rf.predict(X_test)
print("Random Forest")
print("Accuracy:",
      metrics.accuracy_score(Y_test, rf_pred))
print(metrics.classification_report(Y_test, rf_pred))

#Gradient Boosting
gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, Y_train)
gb_pred = gb.predict(X_test)
print("Gradient Boosting")
print("Accuracy:",
      metrics.accuracy_score(Y_test, gb_pred))
print(metrics.classification_report(Y_test, gb_pred))

# Confusion Matrix (Random Forest)
cm = metrics.confusion_matrix(Y_test, rf_pred)
plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest Confusion Matrix")
plt.show()

#Model Comparison
comparison = pd.DataFrame({
    "Model":[
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting"
    ],
    "Accuracy":[
        metrics.accuracy_score(Y_test, log_pred),
        metrics.accuracy_score(Y_test, dt_pred),
        metrics.accuracy_score(Y_test, rf_pred),
        metrics.accuracy_score(Y_test, gb_pred)
    ]
})
print("\nModel Comparison")
print(comparison)

#Accuracy Bar Chart
plt.figure(figsize=(8,5))
sns.barplot(
    x="Accuracy",
    y="Model",
    data=comparison
)
plt.title("Model Comparison")
plt.show()

# Best Model
best_model = comparison.loc[
    comparison["Accuracy"].idxmax()
]
print("Best Performing Model")
print(best_model)