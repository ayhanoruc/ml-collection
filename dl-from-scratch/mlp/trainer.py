import numpy as np
from auto_tensor import AutoTensor

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
    A class that implements basic stochastic gradient descent (SGD) optimization with enhancements for adaptable learning rates and gradient clipping.

    Attributes:
        parameters (list of AutoTensor): The parameters of the model to be optimized.
        lr (float): Learning rate for the optimizer.
        clip_value (float, optional): Maximum allowed value for gradients, used for gradient clipping.
    """

    def __init__(self, parameters, lr=0.01, clip_value=None):
        """
        Initializes the Optimizer with the given parameters and learning rate.

        Parameters:
            parameters (list of AutoTensor): Model parameters to optimize.
            lr (float): Initial learning rate.
            clip_value (float, optional): Threshold for gradient clipping; None disables clipping.
        """
        self.parameters = parameters
        self.lr = lr
        self.clip_value = clip_value
    
    def step(self):
        """
        Performs a single optimization step (parameter update).
        """
        for p in self.parameters:
            if self.clip_value is not None:
                p.grad = np.clip(p.grad, -self.clip_value, self.clip_value)
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


def train(model, data_loader, optimizer, loss_fn, epochs):
    """
    Train the model using batches of data.

    Parameters:
        model: The model to be trained.
        data_loader: Function that yields batches of data.
        optimizer: Optimizer object to adjust model weights.
        loss_fn: Function to compute the loss between predictions and true values.
        epochs (int): Number of complete passes through the dataset.
    """
    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0

        for x_batch, y_batch in data_loader:
            optimizer.zero_grad()

            # Forward pass
            predictions = [model(x) for x in x_batch]
            loss, metric = loss_fn(predictions, y_batch)
            total_loss += loss.value  # Assuming loss is returned as an AutoTensor
            batch_count += 1

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()
        print(f"Epoch {epoch + 1}: metric = {metric}")

        if batch_count > 0:
            average_loss = total_loss / batch_count
            print(f"Epoch {epoch + 1}: Average Loss = {average_loss:.4f}")
        else:
            print(f"Epoch {epoch + 1}: No data processed. Check data loader or batch size.")



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
    correct = np.sum(predictions == labels)
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


