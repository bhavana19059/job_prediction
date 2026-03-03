import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load Dataset
df = pd.read_csv("job_dataset.csv")

# Encoding
le_degree = LabelEncoder()
le_specialization = LabelEncoder()
le_job = LabelEncoder()

df['Degree'] = le_degree.fit_transform(df['Degree'])
df['Specialization'] = le_specialization.fit_transform(df['Specialization'])
df['JobRole'] = le_job.fit_transform(df['JobRole'])

# Features and Target
X = df[['Degree', 'Specialization', 'CGPA']]
y = df['JobRole']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)