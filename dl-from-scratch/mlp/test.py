import unittest
import numpy as np
from auto_tensor import AutoTensor, tanh, relu


class TestAutoTensor(unittest.TestCase):
    def test_addition(self):
        a = AutoTensor(2)
        b = AutoTensor(3)
        c = a + b
        self.assertEqual(c.value, 5, "Addition should compute the correct value.")

    def test_multiplication(self):
        a = AutoTensor(4)
        b = AutoTensor(5)
        c = a * b
        self.assertEqual(c.value, 20, "Multiplication should compute the correct value.")

    def test_activation_tanh(self):
        a = AutoTensor(0.5)
        b = tanh(a)
        expected_value = np.tanh(0.5)
        self.assertAlmostEqual(b.value, expected_value, "Tanh activation function should compute the correct value.")

    def test_activation_relu(self):
        a = AutoTensor(-0.5)
        b = relu(a)
        self.assertEqual(b.value, 0, "ReLU activation should compute the correct value for negative input.")
        a = AutoTensor(0.5)
        b = relu(a)
        self.assertEqual(b.value, 0.5, "ReLU activation should compute the correct value for positive input.")

    def test_backward_pass(self):
        a = AutoTensor(2)
        b = AutoTensor(3)
        c = a * b
        c.grad = 1
        c.backward()
        self.assertEqual(a.grad, 3, "Backward pass should compute correct gradient for 'a'.")
        self.assertEqual(b.grad, 2, "Backward pass should compute correct gradient for 'b'.")

if __name__ == '__main__':
    print(AutoTensor.activations['relu'](AutoTensor(0.5)).value)
    unittest.main()
