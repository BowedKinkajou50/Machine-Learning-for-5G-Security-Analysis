import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Load the dataset
df = pd.read_csv(r'CS-24-327-Machine-Learning-for-5G-Security-Analysis/src/fraud_modified.csv')

# Remove spaces and convert column names to lowercase
df.columns = df.columns.str.replace(' ', '').str.lower()

# Encode the 'class' column
label_encoder = LabelEncoder()
df['isfraud'] = label_encoder.fit_transform(df['isfraud'])

# Split the data into features (X) and target variable (y)
X = df.drop('isfraud', axis=1)
y = df['isfraud']

encoder = OneHotEncoder(drop='first', sparse_output=True)
X_encoded = encoder.fit_transform(X.select_dtypes(include=['object']))

# Convert the encoded features to a DataFrame
X_encoded_df = pd.DataFrame(X_encoded.toarray(), columns=encoder.get_feature_names_out(X.select_dtypes(include=['object']).columns))
X = pd.concat([X.drop(X.select_dtypes(include=['object']).columns, axis=1), X_encoded_df], axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=200)
counter = 1
# Define the parameter grid
param_grid = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [1,2,3, 'sqrt', 'log2', None]
}

# Initialize the Decision Tree classifier
model = DecisionTreeClassifier(random_state=0)

# Perform Grid Search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Print the performance metrics for each set of parameters tried during the grid search
print("Grid Search Results:")
for params, mean_score, std_score in zip(grid_search.cv_results_['params'],
                                          grid_search.cv_results_['mean_test_score'],
                                          grid_search.cv_results_['std_test_score']):
    print("Test: ",counter)
    print("Parameters:", params)
    
    # Fit the model with current parameters and predict
    model.set_params(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate accuracy, precision, and recall
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Print performance metrics
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print()
    counter += 1

# Print the best parameters and performance of the best model
best_params = grid_search.best_params_
print("Best Parameters:")
print(best_params)

best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

accuracy_best = accuracy_score(y_test, y_pred_best)
precision_best = precision_score(y_test, y_pred_best, average='weighted')
recall_best = recall_score(y_test, y_pred_best, average='weighted')

conf_matrix_best = confusion_matrix(y_test, y_pred_best)
print("Confusion Matrix (Best Model):")
print(conf_matrix_best)

print("Best Model Performance:")
print(f"Accuracy: {accuracy_best}")
print(f"Precision: {precision_best}")
print(f"Recall: {recall_best}")
