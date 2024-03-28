import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [1, 2, 3, 4, 5, 'sqrt', 'log2', None]
}

# Load the dataset
df = pd.read_csv(r'CS-24-327-Machine-Learning-for-5G-Security-Analysis/src/ddos_modified.csv')

# Remove spaces and convert column names to lowercase
df.columns = df.columns.str.replace(' ', '').str.lower()

num = df.shape[0]

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=200)

# Initialize the Decision Tree classifier
model = DecisionTreeClassifier(random_state=0)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Create a file to store the results
with open('model_results.txt', 'w') as f:
    f.write("Grid Search Results:\n")

    # Loop through each set of parameters tried during grid search
    for i, (params, mean_score, scores) in enumerate(zip(grid_search.cv_results_['params'], grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['std_test_score'])):
        f.write(f"Test {i+1}:\n")
        f.write("Parameters: {}\n".format(params))

        # Use the parameters to train the model
        model.set_params(**params)
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Evaluate the accuracy, precision, and recall of the model
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted')

        conf_matrix = confusion_matrix(y_test, y_pred)
        f.write("Confusion Matrix:\n")
        f.write("{}\n".format(conf_matrix))

        f.write("Model Performance:\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n\n")

# Print the best model's performance
best_params = grid_search.best_params_
with open('model_results.txt', 'a') as f:
    f.write("Best Parameters:\n")
    f.write("{}\n".format(best_params))

    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)
    y_pred_best = best_model.predict(X_test)

    accuracy_best = accuracy_score(y_test, y_pred_best)
    precision_best = precision_score(y_test, y_pred_best, average='weighted', zero_division=0)
    recall_best = recall_score(y_test, y_pred_best, average='weighted')

    conf_matrix_best = confusion_matrix(y_test, y_pred_best)
    f.write("Confusion Matrix (Best Model):\n")
    f.write("{}\n".format(conf_matrix_best))

    f.write("Best Model Performance:\n")
    f.write(f"Accuracy: {accuracy_best}\n")
    f.write(f"Precision: {precision_best}\n")
    f.write(f"Recall: {recall_best}\n")
