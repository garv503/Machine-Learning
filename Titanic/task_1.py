# Importing the required modules
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Loading the dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Data Preprocessing
df.fillna(method='ffill', inplace=True)  # Handling missing values

# Encode categorical variables
# Convert 'Sex' column to numerical values (0 for female, 1 for male)
df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})

# One-hot encode 'Embarked' column
df = pd.get_dummies(df, columns=['Embarked'])

# Dropping irrelevant columns
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Separating the data into x and y
x = df.drop('Survived', axis=1)
y = df['Survived']

# Separating the data into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

# Model Building
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(x_train, y_train)

# Evaluating the model
# Predict on test set
y_pred = clf.predict(x_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))