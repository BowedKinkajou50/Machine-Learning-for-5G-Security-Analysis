#Authors: Miles Fagan and James McGlone.
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

#works with fruit data

# Load the dataset
data = pd.read_csv(r'src\fruit_data.csv')

# Encode categorical variables
le = LabelEncoder()
data['Color'] = le.fit_transform(data['Color'])
data['Type'] = le.fit_transform(data['Type'])

# Split the data into features and target
X = data[['Color']]
y = data['Type']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Multi-layer Perceptron Classifier
clf = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000)

# Train the classifier
clf.fit(X_train, y_train)

# Predictions on the test set
predictions = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# Calculate precision
precision = precision_score(y_test, predictions, average='weighted')
print("Precision:", precision)

# Calculate recall
recall = recall_score(y_test, predictions, average='weighted')
print("Recall:", recall)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(conf_matrix)
