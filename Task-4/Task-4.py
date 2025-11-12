import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv("D:/Work/ML Tasks/Titanic-Dataset.csv")
print("Sample records from Dataset:\n")
print(df.head())

# Clean the data
df.fillna({'Age': df['Age'].median(), 'Embarked': df['Embarked'].mode()[0]}, inplace=True)

label = LabelEncoder()
df['Sex'] = label.fit_transform(df['Sex'])
df['Embarked'] = label.fit_transform(df['Embarked'])

# Visualize dataset
sns.countplot(x='Survived', data=df)
plt.title("Survival Count (0 = Not Survived, 1 = Survived)")
plt.show()

sns.barplot(x='Sex', y='Survived', data=df)
plt.title("Gender vs Survival\n(0 = Female, 1 = Male)")
plt.show()

X = df[['Pclass','Sex','Age','SibSp','Parch','Fare']]
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a Logistic Model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("Accuracy:", (acc * 100),"%")

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n",cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,  display_labels=['Not Survived', 'Survived'])
disp.plot(cmap=plt.cm.Greens)
plt.title("Confusion Matrix - Logistic")

plt.show()
