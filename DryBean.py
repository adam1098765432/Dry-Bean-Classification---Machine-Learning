import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("Dry_Bean_Dataset.csv")

# As in the example, convert integer columns to float for consistency
int_columns = df.select_dtypes(include=["int64"]).columns
df[int_columns] = df[int_columns].astype("float64")

# (If the dataset has more than 10,000 rows, sample 10,000 randomly)
if df.shape[0] > 10000:
    df = df.sample(n=10000, random_state=33)

# Separate features and label.
# Assume the label is the last column.
X = df.iloc[:, :-1].to_numpy()
y = df.iloc[:, -1].to_numpy()

# Standardize features, similar to the approach in the examples
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and test sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Set up a grid search for SVC with RBF kernel using code found in other examples
### Create a support vector classifier using radial basis functions for your kernel
param_grid = {
    'C': [0.1, 1, 10, 50, 100],
    'gamma': [0.001, 0.01, 0.1, 0.125, 1]
}

# Create the SVC model (using one-versus-one decision function as in the examples)
svc = SVC(kernel='rbf', decision_function_shape='ovo')

### grid search over the parameter grid to find the optimal C and gamma
grid_search = GridSearchCV(svc, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Prints the best C and gamma and cross-validation score
print("Best parameters found:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# Retrieve the best estimator and evaluate its performance
best_svc = grid_search.best_estimator_

# Evaluate on training set
y_train_pred = best_svc.predict(X_train)
print(f"Train Accuracy: {accuracy_score(y_train, y_train_pred):.3f}")

# Evaluate on test set
y_test_pred = best_svc.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.3f}")
print(f"Number of support vectors: {best_svc.support_vectors_.shape[0]}")


### Generate and display the confusion matrix (using the same approach as in the examples)
cm = confusion_matrix(y_test, y_test_pred)
disp_cm = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_svc.classes_)
disp_cm.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for SVC with RBF Kernel")
plt.show()


# When I ran it, the train accuracy was slightly higher than the test accuracy meaning slight overfitting