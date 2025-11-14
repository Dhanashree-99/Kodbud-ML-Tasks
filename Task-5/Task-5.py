import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv("D:/Work/ML Tasks/loan_approval_dataset.csv")
print("Sample records:\n", df.head())

# Handle missing values
for col in df.columns:
    if df[col].dtype == 'object':   # for categorical columns
        df[col] = df[col].fillna(df[col].mode()[0])
    else:                           # for numeric columns
        df[col] = df[col].fillna(df[col].median())

print("\nMissing values after handling:\n",df.isnull().sum())

# Encoding
encoder = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = encoder.fit_transform(df[col])

X = df.drop('loan_status', axis=1)
y = df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
predicted_labels = encoder.inverse_transform(y_pred)
print("\nFrist 10 sample Predicted Loan Status:\n", predicted_labels[:10])

print(f"\nModel Accuracy: {acc*100:.2f}%")

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Rejected', 'Approved'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix - Loan Approval Prediction")
plt.show()
