import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from joblib import Parallel, delayed

# Load the dataset
df = pd.read_csv(r'/Users/megansorochin/Desktop/csv/Fraud.csv')

# Remove spaces and convert column names to lowercase
df.columns = df.columns.str.replace(' ', '').str.lower()

# Drop the 'step' column for simplicity
df = df.drop('step', axis=1)

start_index = 500000
end_index = 6362619

df = df.drop(index=range(start_index, end_index + 1), inplace=True)

num_rows = df.shape[0]
# Print the number of rows
print(f"The dataset has {num_rows} rows.")

# Encode the 'class' column
label_encoder = LabelEncoder()
df['isfraud'] = label_encoder.fit_transform(df['isfraud'])

# Split the data into features (X) and target variable (y)
X = df.drop('isfraud', axis=1)
y = df['isfraud']

# Use OneHotEncoder to encode categorical variables as sparse matrix
encoder = OneHotEncoder(drop='first', sparse_output=True)
X_encoded = encoder.fit_transform(X.select_dtypes(include=['object']))

# Convert the encoded features to a DataFrame
X_encoded_df = pd.DataFrame.sparse.from_spmatrix(X_encoded, columns=encoder.get_feature_names_out(X.select_dtypes(include=['object']).columns))
X = pd.concat([X.drop(X.select_dtypes(include=['object']).columns, axis=1), X_encoded_df], axis=1)

# Split the data in parallel
split_results = Parallel(n_jobs=-4, timeout = 1000)(delayed(train_test_split)(X, y, test_size=0.2, random_state=42) for _ in range(10))

# Flatten the results
X_train, X_test, y_train, y_test = map(np.concatenate, zip(*split_results))

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