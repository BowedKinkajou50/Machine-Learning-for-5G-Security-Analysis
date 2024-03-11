import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
df = pd.read_csv(r'src/ddos_modified.csv')

# Remove spaces and convert column names to lowercase
df.columns = df.columns.str.replace(' ', '').str.lower()

num = df.shape[0]
print(num)
# Encode the 'label' column
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

# Split the data into features (X) and target variable (y)
X = df.drop('label', axis=1)
y = df['label']

encoder = OneHotEncoder(drop='first', sparse_output=True)
X_encoded = encoder.fit_transform(X.select_dtypes(include=['object']))


# Convert the encoded features to a DataFrame
X_encoded_df = pd.DataFrame(X_encoded.toarray(), columns=encoder.get_feature_names_out(X.select_dtypes(include=['object']).columns))
X = pd.concat([X.drop(X.select_dtypes(include=['object']).columns, axis=1), X_encoded_df], axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=500)

# Initialize the Naive Bayes classifier
model = GaussianNB()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the accuracy, precision, and recall of the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')  # Use average='weighted' for multi-class
recall = recall_score(y_test, y_pred, average='weighted')  # Use average='weighted' for multi-class

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")