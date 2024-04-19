import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from neural_structures import MLP
from auto_tensor import AutoTensor
from trainer import SGDOptimizer, Loss, data_loader, train, evaluate, accuracy, predict, mse_loss,\
                    train_and_validate

from random import random

np.random.seed(42)

# Function to generate a sample learnable dataset
def generate_normalized_data(num_samples, num_features, num_classes):
    X = np.random.randn(num_samples, num_features)  # Random features
    # Normalize the features to have mean 0 and variance 1
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    true_weights = np.random.randn(num_features, num_classes)  # Random weights for our model
    noise = 0.1 * np.random.randn(num_samples, num_classes)  # Some noise
    
    # Generate targets: multiply features with weights and add noise
    Y = X.dot(true_weights) + noise
    
    # Apply a threshold to get binary class labels
    Y = np.where(Y > 0, 1, -1)
    
    return X, Y.squeeze()

# Generate a sample dataset with 5 features, 20 samples, and binary class labels
X_sample, Y_sample = generate_normalized_data(40, 5, 1)

X_train, y_train, X_val, y_val  = X_sample.tolist()[:30], Y_sample.tolist()[:30], X_sample.tolist()[30:], Y_sample.tolist()[30:]




mlp = MLP(5, [8, 8, 1], ['relu', 'relu','tanh'])
n_iters = 1000
learning_rate = 0.01

for i in range(n_iters):
    # forward pass
    y_preds = [mlp(x) for x in X_train]
    
    # compute loss
    loss = sum((y-y_pred) ** 2 for y_pred, y in zip(y_preds, y_train))/len(y_train)
    
   
    # reset gradients
    mlp.zero_grad()

    # backward pass
    loss.backward()

    # update
    for p in mlp.parameters():
        p.value += -learning_rate * p.grad
    
    if i % 100 == 0:
        print(f'Iteration: {i}, Loss: {loss.value}')

    # apply early stopping
    if loss.value < 0.1:
        print(f'Iteration: {i}, Loss: {loss.value}')
        break


mlp = MLP(5, [8, 8, 1], ['relu', 'relu','tanh'])
optimizer = SGDOptimizer(mlp.parameters(), lr=0.1)
epochs=1000
eval_every = 100
print("using trainer method")
train_and_validate(mlp, X_train, y_train, X_val, y_val, optimizer, mse_loss,accuracy,epochs,eval_every)
#train(mlp, X_train, y_train, optimizer, mse_loss,accuracy,epochs)