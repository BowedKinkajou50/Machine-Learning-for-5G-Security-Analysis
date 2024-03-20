#Authors: Miles Fagan and James McGlone.
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

#now works with ddos modified


# Load the dataset
data = pd.read_csv(r'src/ddos_modified.csv')

# Encode categorical variables
le = LabelEncoder()
data['label'] = le.fit_transform(data['label'])

# Split the data into features and target
X = data[['totalfwdpackets', 'totalbackwardpackets', 'totallengthoffwdpackets', 'totallengthofbwdpackets']]
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=500)

# Initialize the Multi-layer Perceptron Classifier
clf = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000)

# Train the classifier
clf.fit(X_train, y_train)

# Predictions on the test set
predictions = clf.predict(X_test)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(conf_matrix)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# Calculate precision
precision = precision_score(y_test, predictions, average='weighted')
print("Precision:", precision)

# Calculate recall
recall = recall_score(y_test, predictions, average='weighted')
print("Recall:", recall)
