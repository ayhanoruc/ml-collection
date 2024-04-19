import numpy as np 



class AutoTensor:
    """
    Represents a differentiable tensor in a computational graph that supports automatic differentiation.
    """
    def __init__(self, value, _children=(), _ops='', label=''):
        """
        Initialize an AutoTensor object.
        
        Parameters:
            value (float): The numeric value of this tensor.
            _children (tuple): Nodes from which this tensor was derived.
            _ops (str): Operation that produced this tensor.
            label (str): Optional label for identifying the tensor.
        """
        self.value = value
        self._prev = set(_children)
        self._op = _ops
        self.label = label
        self.grad = 0.0
        self._backward = lambda: None  # Initialize as function; no operation for leaf nodes.

    def __repr__(self) -> str:
        """
        Return a string representation of the AutoTensor.
        """
        return f"AutoTensor(value={self.value}, name={self.label})"

    def __add__(self, other):
        """
        Perform element-wise addition of two tensors.
        """
        other = other if isinstance(other, AutoTensor) else AutoTensor(other)
        out = AutoTensor(self.value + other.value, (self, other), '+')

        def backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = backward
        return out

    def __radd__(self, other):
        """
        Support right addition for non-AutoTensor objects.
        """
        return self + other

    def __mul__(self, other):
        """
        Perform element-wise multiplication of two tensors.
        """
        other = other if isinstance(other, AutoTensor) else AutoTensor(other)
        out = AutoTensor(self.value * other.value, (self, other), '*')

        def backward():
            self.grad += other.value * out.grad
            other.grad += self.value * out.grad
        out._backward = backward
        return out

    def __rmul__(self, other):
        """
        Support right multiplication for non-AutoTensor objects.
        """
        return self * other

    def __truediv__(self, other):
        """
        Perform element-wise division of this tensor by another tensor or a scalar.
        """
        if not isinstance(other, AutoTensor):
            other = AutoTensor(other)  # Convert scalar to AutoTensor for uniform handling
        if other.value == 0:
            raise ValueError("Cannot divide by zero")

        out = AutoTensor(self.value / other.value, (self, other), '/')
    
        def backward():
            # Apply the chain rule for division
            self.grad += (1 / other.value) * out.grad
            other.grad -= (self.value / (other.value ** 2)) * out.grad
        out._backward = backward
        
        return out


    def __rtruediv__(self, other):
        """
        Support right division for non-AutoTensor objects. This is other / self.
        """
        if not isinstance(other, AutoTensor):
            other = AutoTensor(other)  # Convert scalar to AutoTensor if necessary
        
        if self.value == 0:
            raise ValueError("Cannot divide by zero")

        out = AutoTensor(other.value / self.value, (other, self), '/')

        def backward():
            # Apply the chain rule for division in the reverse order
            self.grad -= (other.value / (self.value ** 2)) * out.grad
            other.grad += (1 / self.value) * out.grad
            
        out._backward = backward

        return out

    def __pow__(self, other):
        """
        Raise tensor to the power of 'other'.
        """
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = AutoTensor(self.value ** other, (self,), f'**{other}')

        def backward():
            self.grad += (other * self.value ** (other - 1)) * out.grad
        out._backward = backward
        return out

    def __neg__(self):
        """
        Negate the tensor.
        """
        return self * -1

    def __sub__(self, other):
        """
        Perform element-wise subtraction of two tensors.
        """
        return self + (-other)

    def __rsub__(self, other):
        """
        Support right subtraction for non-AutoTensor objects.
        """
        return other + (-self)
    
    def clip(self, min_value, max_value):
        # Ensure that the value is clipped according to the boundaries provided
        self.value = max(min(self.value, max_value), min_value)
        return self


    def tanh(self):
        """
        Compute the hyperbolic tangent,  nonlinear activation function, of the tensor.
        """
        x = self.value
        t = np.tanh(x) # numpy's tanh is used here for stability
        out = AutoTensor(t, (self,), 'tanh')

        def backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = backward
        return out
    
    def relu(self):
        """
        Compute the rectified linear unit activation function to introduce non-linearity.
        """
        out = AutoTensor(max(0, self.value), (self, ), 'relu')
        def backward():
            self.grad += (out.value > 0) * out.grad
        out._backward = backward
        return out

    def backward(self):
        """
        Perform backpropagation to compute gradients for all tensors in the graph,
        using topological sort to ensure that all tensors are computed in the correct order.
        """
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


    @classmethod
    def get_activation_function(cls, name: str):
        """
        Retrieves an activation function by name, using the internal implementations within AutoTensor class.

        Parameters:
            name (str): The name of the activation function to retrieve.

        Returns:
            Callable: The activation function as a method bound to AutoTensor instances.

        Raises:
            AttributeError: If no such function exists in the AutoTensor.
        """
        # Directly return the method bound to the class, which will be called on an instance
        try:
            # Here we access the instance method directly through the class.
            function = getattr(AutoTensor, name)
        except AttributeError:
            raise ValueError(f"No activation function named '{name}' found in AutoTensor.")
        return function
