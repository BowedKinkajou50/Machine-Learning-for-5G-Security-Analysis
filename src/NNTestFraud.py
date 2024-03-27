import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Works for fraud_modified and runs different test values

# Load the dataset
data = pd.read_csv(r'src/fraud_modified.csv')

# Encode categorical variables
le = LabelEncoder()
data['type'] = le.fit_transform(data['type'])

# Split the data into features and target
X = data[['amount']]
y = data['isfraud']

# Define ranges for test sizes, random states, and MLP classifier configurations
test_sizes = [0.1, 0.2, 0.5, 0.8]  # Example test sizes
random_states = [42, 200, 500]  # Example random states
layer_size_configs = [(100,), (50, 50), (200,)]  # Example MLP configurations
test_number= 1
# Iterate over test sizes
for test_size in test_sizes:
    # Iterate over random states
    for random_state in random_states:
        # Iterate over MLP configurations
        for mlp_config in layer_size_configs:
            print(f"Test Number: {test_number}.\nTesting with test_size={test_size}, random_state={random_state}, hidden_layer_sizes= {mlp_config}")
            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            # Initialize the Multi-layer Perceptron Classifier
            clf = MLPClassifier(hidden_layer_sizes=mlp_config, max_iter=1000)

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
            precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
            print("Precision:", precision)

            # Calculate recall
            recall = recall_score(y_test, predictions, average='weighted')
            print("Recall:", recall)
            print("\n")
            test_number= test_number+1

