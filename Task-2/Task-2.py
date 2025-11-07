import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

iris = load_iris()
X = iris.data   
y = iris.target 

data = pd.DataFrame(X, columns=iris.feature_names)
data['species'] = iris.target_names[y]

print("First 5 rows of the dataset:")
print(data.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# For KNN Model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

# Evaluation for KNN
acc_knn = accuracy_score(y_test, y_pred_knn)
print("\n* KNN Model Results *")
print("Accuracy:", (acc_knn * 100), "%")
print("\nClassification Report (KNN):")
print(classification_report(y_test, y_pred_knn, target_names=iris.target_names))

print("Confusion Matrix (KNN):")
print(confusion_matrix(y_test, y_pred_knn))

# Confusion Matrix for KNN
cm = confusion_matrix(y_test, y_pred_knn)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap=plt.cm.Purples)
plt.title("Confusion Matrix - KNN")
plt.show()

# For Logistic Model
log_model = LogisticRegression(max_iter=200)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

# Evaluation for Logistic Regression
acc_log = accuracy_score(y_test, y_pred_log)
print("\n* Logistic Regression Results *")
print("Accuracy:", (acc_log * 100), "%")
print("\nClassification Report (Logistic Regression):")
print(classification_report(y_test, y_pred_log, target_names=iris.target_names))

print("Confusion Matrix (Logistic):")
print(confusion_matrix(y_test, y_pred_log))

# Confusion Matrix for Logistic Regression
cm = confusion_matrix(y_test, y_pred_log)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap=plt.cm.Greens)
plt.title("Confusion Matrix - Logistic")
plt.show()

print("\n~ Model Accuracy Comparison:")
print(f"= KNN Accuracy: {acc_knn*100}%")
print(f"= Logistic Regression Accuracy: {acc_log*100}%")

if acc_knn > acc_log:
    print("\n~ KNN performed slightly better on this dataset.\n")
elif acc_log > acc_knn:
    print("\n~ Logistic Regression performed slightly better on this dataset.\n")
else:
    print("\n~ Both models performed equally well!\n")
