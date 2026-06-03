# ================================
# Logistic Regression Pipeline
# ================================

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def plot_decision_boundary(model, X, y):
    # Create a mesh grid for plotting
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # Predict the function value for the whole grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.title('Logistic Regression Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()


data = load_breast_cancer()

X = data.data       # Features
y = data.target     # Labels

# Split the dataset 
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Feature Scaling

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train) 
X_test = scaler.transform(X_test)


# Create Logistic Regression model

model = LogisticRegression()

# train the model
model.fit(X_train, y_train)

# make predictions
y_pred = model.predict(X_test)

# Probability predictions
y_prob = model.predict_proba(X_test)


# Plot decision boundary (using first two features for visualization)
plot_decision_boundary(model, X_train[:, :2], y_train)


# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Custom Prediction
sample = X_test[0].reshape(1, -1)

prediction = model.predict(sample)

probability = model.predict_proba(sample)

print("\nPrediction:", prediction)
print("Probability:", probability)