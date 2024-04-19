
import random
from auto_tensor import AutoTensor
from typing import List



class Module:
    """
    Base class for all modules in the neural network.
    Provides utility methods to zero out gradients and retrieve parameters.
    """
    
    def zero_grad(self):
        """
        Resets the gradients of all parameters in the module to zero.
        """
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        """
        Should be overridden by subclasses to return a list of parameters (AutoTensor objects) that require gradients.

        Returns:
            list: A list of parameters to be used in optimization and gradient computation.
        """
        return []
    


class Neuron(Module):
    """
    Represents a neuron in a neural network, with associated weights and bias.
    This class performs the forward pass of a neuron using a configurable activation function.
    """
    def __init__(self, nin: int, activation: str = 'tanh'):
        """
        Initialize a neuron with a specified number of input connections and an activation function.
        
        Parameters:
            nin (int): The number of inputs the neuron should accept.
            activation (str): The name of the activation function to use, specified as a string.
        """
        self.w = [AutoTensor(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = AutoTensor(random.uniform(-1, 1))
        self.activation = AutoTensor.get_activation_function(activation)
    
    def __call__(self, x: List[AutoTensor]) -> AutoTensor:
        """
        Perform the forward pass using the inputs provided and apply the activation function.
        
        Parameters:
            x (list of AutoTensor): A list of input values, each as an AutoTensor.
        
        Returns:
            AutoTensor: The output of the neuron after applying the activation function.
        """
        if len(self.w) != len(x):
            raise ValueError(f"expected {len(self.w)} inputs, got {len(x)}")

        out = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return self.activation(out)
    
    def parameters(self) -> List[AutoTensor]:
        """
        Get the parameters of the neuron, including its weights and bias.
        """
        return self.w + [self.b]
    
    def __repr__(self):
        """
        Return a string representation of the neuron.
        """
        return f"Neuron(nin={len(self.w)}, bias={self.b})"



class DenseLayer(Module):
    """
    Represents a dense layer in a neural network, each neuron in the layer is connected to all inputs.
    """
    def __init__(self, nin, nout, activation: str = 'tanh'):
        """
        Initialize a DenseLayer with a specified number of input and output neurons, and an activation function.
        
        Parameters:
            nin (int): The number of inputs each neuron in the layer should accept.
            nout (int): The number of neurons in this layer.
            activation (str): The name of the activation function to use in each neuron, specified as a string.
        """
        self.neurons = [Neuron(nin, activation) for _ in range(nout)]
        self.nin = nin
        self.nout = nout

    def __call__(self, x) -> List[AutoTensor]:
        """
        Perform the forward pass by passing the input through each neuron in the layer.
        """
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self) -> List[AutoTensor]:
        """
        Retrieve all trainable parameters from all neurons in the layer.
        """
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params

    def __repr__(self):
        return f"DenseLayer(nin={self.nin}, nout={self.nout}, activation={self.activation.__name__})"
    





class MLP(Module):
    """
    Represents a multi-layer perceptron (MLP), a basic form of a neural network.
    
    Attributes:
        nin (int): The number of input features.
        nouts (list of int): The number of neurons in each layer of the network.
    
    Parameters:
        nin (int): The dimensionality of the input data.
        nouts (list of int): A list of output sizes for each layer.
        activations (str or list of str): The activation function(s) to be used for each layer. 
                                        Can be a single string (applied to all layers) or a list of strings.
    """
    def __init__(self, nin, nouts, activations='relu'):
        sz = [nin] + nouts  # This will produce a list of sizes of each layer.
        # Determine activation functions for each layer
        if isinstance(activations, str):
            activations = [activations] * len(nouts)
        elif len(activations) != len(nouts):
            raise ValueError("Length of activations must match the length of nouts.")
        
        self.layers = [DenseLayer(sz[i], sz[i+1], activation=activations[i]) for i in range(len(nouts))]

    def __call__(self, x):
        """
        Forward pass through the MLP.
        
        Parameters:
            x (list of AutoTensor): Input data.
        
        Returns:
            AutoTensor: The output of the network.
        """
        for layer in self.layers:
            x = layer(x)  # Apply each layer sequentially
        return x

    def parameters(self):
        """
        Retrieve all trainable parameters from all layers of the network.
        
        Returns:
            list of AutoTensor: A list containing all weights and biases from the network.
        """
        return [p for layer in self.layers for p in layer.parameters()]