import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from neural_structures import MLP
from trainer import SGDOptimizer, Loss, data_loader, train, evaluate, accuracy, predict
from random import random

# Generate synthetic data
X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
y = y * 2 - 1  # Convert labels to -1 or 1

# Plot the data
plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
plt.title("Synthetic 'make_moons' Dataset")
plt.show()


# Assuming your MLP and other classes are defined as in your reference.
np.random.seed(1337)

# Data Preparation
X, y = make_moons(n_samples=100, noise=0.1)
y = y * 2 - 1  # Convert y to -1 or 1
dataset = (X, y)

# Model Initialization
model = MLP(2, [16, 16, 1], ['relu', 'relu', 'tanh'])


# Prepare data loader
batch_size = 32
train_loader = data_loader(dataset, batch_size=batch_size, shuffle=True)

# Define the optimizer
optimizer = SGDOptimizer(parameters=model.parameters(), lr=0.01)

# Define a simple loss function for binary classification
def loss_fn(y_pred, y_true):
    # Assuming y_pred and y_true are lists of AutoTensor and actual labels respectively
    # Calculate binary cross-entropy loss
    bce_loss = Loss.binary_cross_entropy(y_pred, y_true)
    
    # Calculate accuracy as a single float value
    accuracy = sum((yp.value > 0) == (yt == 1) for yp, yt in zip(y_pred, y_true)) / len(y_true)
    return bce_loss, accuracy

# Train the model
epochs = 400
train(model, train_loader, optimizer, loss_fn, epochs)

# Define the metric as accuracy
def metric(predictions, labels):
    return accuracy(np.round(predictions), labels)

# Evaluate the model
print("Evaluating the model...")
eval_loader = data_loader(dataset, batch_size=len(X), shuffle=False)  # Use the entire dataset for evaluation in one go
eval_accuracy = evaluate(model, eval_loader, metric)
print(f"Model accuracy: {eval_accuracy * 100:.2f}%")


# make inference
# Make predictions on new data
new_data = np.array([[1.5, -0.5], [-1.0, 0.5]])  # Example new data points
predictions = predict(model, new_data)
print("Predictions:", predictions)