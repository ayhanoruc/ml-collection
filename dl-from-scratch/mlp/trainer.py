import numpy as np
from auto_tensor import AutoTensor
import matplotlib.pyplot as plt
from typing import Callable
class Loss:
    """
    A class containing various loss functions and their derivatives for training neural networks.
    """

    @staticmethod
    def mse(y_pred, y_true):
        """
        Calculate the mean squared error between predictions and true values.

        Parameters:
            y_pred (np.array): Predicted values.
            y_true (np.array): True values.

        Returns:
            float: The mean squared error.
        """
        return np.mean((y_pred - y_true) ** 2)
    
    @staticmethod
    def mse_grad(y_pred, y_true):
        """
        Calculate the gradient of the mean squared error loss with respect to predictions.

        Parameters:
            y_pred (np.array): Predicted values.
            y_true (np.array): True values.

        Returns:
            np.array: Gradient of the mean squared error.
        """
        return 2 * (y_pred - y_true) / y_true.size

    @staticmethod
    def binary_cross_entropy(y_pred, y_true):
        """
        Compute binary cross-entropy loss.

        Parameters:
            y_pred (np.array): Predicted probabilities (between 0 and 1).
            y_true (np.array): Actual binary labels (0 or 1).

        Returns:
            float: Binary cross-entropy loss.
        """
        epsilon = 1e-15
        y_pred = [yp.clip(epsilon, 1 - epsilon) for yp in y_pred]  # Use the new clip method
        losses = [-yt * np.log(yp.value) - (1 - yt) * np.log(1 - yp.value) for yp, yt in zip(y_pred, y_true)]
        return sum(losses) / len(losses)

    @staticmethod
    def binary_cross_entropy_grad(y_pred, y_true):
        """
        Compute the gradient of the binary cross-entropy loss with respect to predictions.

        Parameters:
            y_pred (np.array): Predicted probabilities (between 0 and 1).
            y_true (np.array): Actual binary labels (0 or 1).

        Returns:
            np.array: Gradient of the binary cross-entropy loss.
        """
        epsilon = 1e-15
        y_pred = [yp.clip(epsilon, 1 - epsilon) for yp in y_pred]
        grads = [(yp.value - yt) / (yp.value * (1 - yp.value) * len(y_true)) for yp, yt in zip(y_pred, y_true)]
        return grads


class SGDOptimizer:
    """
    A class that implements basic stochastic gradient descent (SGD) optimization.

    Attributes:
        parameters (list of AutoTensor): The parameters of the model to be optimized.
        lr (float): Learning rate for the optimizer.
    """

    def __init__(self, parameters, lr=0.01):
        """
        Initializes the Optimizer with the given parameters and learning rate.

        Parameters:
            parameters (list of AutoTensor): Model parameters to optimize.
            lr (float): Initial learning rate.
        """
        self.parameters = parameters
        self.lr = lr
    
    def step(self):
        """
        Performs a single optimization step (parameter update).
        """
        for p in self.parameters:
            p.value -= self.lr * p.grad
    
    def zero_grad(self):
        """
        Resets all gradients to zero, preparing for the next update cycle.
        """
        for p in self.parameters:
            p.grad = 0


def data_loader(dataset, batch_size, shuffle=True, seed=None):
    """
    Generator to yield batches of data, where each data point is converted to a Value object.

    Parameters:
        dataset (list or array-like): The complete dataset to load in batches.
        batch_size (int): The number of data points in each batch.
        shuffle (bool, optional): Whether to shuffle the dataset before creating batches.
        seed (int, optional): Random seed for reproducibility if shuffle is True.

    Yields:
        tuple: Two lists, each containing Value objects, one for x values and the other for y values.
    """
    if not dataset:
        raise ValueError("Dataset is empty.")
    if batch_size <= 0:
        raise ValueError("Batch size must be a positive integer.")

    X, y = dataset
    dataset_size = len(X)
    indices = list(range(dataset_size))
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)
    
    for start_idx in range(0, dataset_size, batch_size):
        batch_indices = indices[start_idx:start_idx + batch_size]
        x_batch = [list(map(AutoTensor, X[i])) for i in batch_indices]
        y_batch = [AutoTensor(y[i]) for i in batch_indices]
        yield x_batch, y_batch





def mse_loss(predictions, targets):
    mse = sum((y-y_pred) ** 2 for y_pred, y in zip(predictions, targets))/len(targets)
    return mse


import matplotlib.pyplot as plt

def train_and_validate(model, X_train, y_train, X_val, y_val, optimizer, loss_fn:Callable, metric:Callable, epochs, eval_every:int=100):
    """
    Train and validate the model using the entire dataset.

    Parameters:
        model: The model to be trained and validated.
        X_train: List of training samples.
        y_train: List of labels corresponding to the training samples.
        X_val: List of validation samples.
        y_val: List of labels corresponding to the validation samples.
        optimizer: Optimizer object to adjust model weights.
        loss_fn: Function to compute the loss between predictions and true values.
        metric: Function to compute the accuracy or other performance metrics.
        epochs (int): Number of complete passes through the dataset.
        eval_every (int): Frequency of epochs to run validation.
    """
    train_losses = []
    val_losses = []
    train_metrics = []
    val_metrics = []

    for epoch in range(epochs):
        # Forward pass on training data
        y_preds_train = [model(x) for x in X_train]
        loss_train = loss_fn(y_preds_train, y_train)
        metric_train = metric(y_preds_train, y_train)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        train_losses.append(loss_train.value)
        train_metrics.append(metric_train)



        # Validation phase
        if epoch % eval_every == 0:
            y_preds_val = [model(x) for x in X_val]
            loss_val = loss_fn(y_preds_val, y_val)
            metric_val = metric(y_preds_val, y_val)
            val_losses.append(loss_val.value)
            val_metrics.append(metric_val)
            print(f"Epoch {epoch + 1}: Train Loss = {loss_train.value:.4f}, Train Metric = {metric_train:.4f}, Validation Loss = {loss_val.value:.4f}, Validation Metric = {metric_val:.4f}")
            # apply early stopping
            if loss_val.value < 0.1:
                print(f'Iteration: {epoch}, Loss: {loss_val.value}')
                break
        else:
            print(f"Epoch {epoch + 1}: Train Loss = {loss_train.value:.4f}, Train Metric = {metric_train:.4f}")

        
    plot_history(train_losses, val_losses, train_metrics, val_metrics, eval_every)

    return train_losses, val_losses, train_metrics, val_metrics



def plot_history(train_losses, val_losses, train_metrics, val_metrics, eval_every, title='Training and Validation Performance'):
    """
    Plot the training and validation loss and metrics over epochs.

    Parameters:
        train_losses (list of float): List of training losses per epoch.
        val_losses (list of float): List of validation losses at specified intervals.
        train_metrics (list of float): List of training metrics per epoch.
        val_metrics (list of float): List of validation metrics at specified intervals.
        eval_every (int): Frequency of epochs at which validation was performed.
        title (str): Title for the plot.
    """
    epochs = len(train_losses)  # Total number of epochs
    validation_epochs = range(0, epochs, eval_every)  # Epochs at which validation was performed

    plt.figure(figsize=(12, 6))

    # Plot for losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(validation_epochs, val_losses, label='Validation Loss', linestyle='--')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot for metrics
    plt.subplot(1, 2, 2)
    plt.plot(train_metrics, label='Train Metric')
    plt.plot(validation_epochs, val_metrics, label='Validation Metric', linestyle='--')
    plt.title('Metric over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.legend()

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the rect so the title fits without overlap
    plt.savefig('dl-from-scratch/mlp/mlp-training-plot.png')
    #plt.show()
    

def train(model, X_train, y_train, optimizer, loss_fn,metric, epochs):
    """
    Train the model using the entire dataset.

    Parameters:
        model: The model to be trained.
        X_train: List of training samples.
        y_train: List of labels corresponding to the training samples.
        optimizer: Optimizer object to adjust model weights.
        loss_fn: Function to compute the loss between predictions and true values.
        epochs (int): Number of complete passes through the dataset.
    """
    for epoch in range(epochs):
        
        # Forward pass
        y_preds = [model(x) for x in X_train]
        #loss = sum((y-y_pred) ** 2 for y_pred, y in zip(y_preds, y_train))
        loss = mse_loss(y_preds, y_train)
        print(loss)
        metric_score = metric(y_preds, y_train)
        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()
        #for p in model.parameters():
        #    p.value += -0.01 * p.grad

        # Reset gradients
        optimizer.zero_grad()

        # Output the loss and metric for this epoch
        print(f"Epoch {epoch + 1}: Loss = {loss.value:.4f}, Metric = {metric_score:.4f}")



def evaluate(model, data_loader, metric):
    total_metric = 0
    batch_count = 0
    for x_batch, y_batch in data_loader:
        predictions = [model(x) for x in x_batch]
        metric_value = metric(predictions, y_batch)
        total_metric += metric_value
        batch_count += 1
    
    if batch_count > 0:
        final_metric = total_metric / batch_count
        print(f"Evaluated Metric: {final_metric}")
        return final_metric
    else:
        print("No data was processed during evaluation.")
        return None


def accuracy(predictions, labels):
    """
    Calculates accuracy as the proportion of correct predictions.

    Parameters:
        predictions (np.array): The predicted labels.
        labels (np.array): The true labels.

    Returns:
        float: The accuracy of predictions.
    """
    pred_values = [p.value for p in predictions]

    # Convert predictions to binary outcomes based on the threshold, 0 for tanh, 0.5 for sigmoid
    preds = np.where(np.array(pred_values) > 0, 1, -1)
    correct = np.sum(preds == np.array(labels))
    total = len(labels)
    return correct / total


def rmse(y_pred, y_true):
        """
        Calculate the root mean squared error between predictions and true values.

        Parameters:
            y_pred (np.array): Predicted values.
            y_true (np.array): True values.
        
        Returns:
            float: The root mean squared error.
        """

        return np.sqrt(np.mean((y_pred - y_true) ** 2))



def predict(model, x):
    """
    Generates predictions from the model for the provided input.

    Parameters:
        model: The trained model used for making predictions.
        x (array-like or iterable): The input data for which predictions are to be made. Can be a single data point or a batch of data points.

    Returns:
        array-like: The predicted outputs from the model.
    """
    try:
        # Assuming the model can handle array-like inputs directly
        predictions = model(x)
    except Exception as e:
        # Logging the exception can be more sophisticated depending on the setup
        print(f"An error occurred during prediction: {e}")
        return None
    
    # Post-processing of predictions could be added here if necessary

    return predictions


