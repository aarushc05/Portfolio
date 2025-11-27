"""
Neural Network Implementation from Scratch

A two-layer fully connected neural network implementation using NumPy.
Features SiLU activation, softmax output, dropout regularization,
and support for both SGD and Adam optimizers.

Author: Aarush Chhiber
"""

import numpy as np


class NeuralNet:
    """
    A two-layer fully connected neural network for multi-class classification.
    
    Architecture: Input -> Hidden1 (SiLU) -> Hidden2 (SiLU) -> Output (Softmax)
    
    Features:
        - SiLU (Swish) activation function
        - Dropout regularization
        - Adam optimizer support
        - Mini-batch gradient descent
    
    Attributes:
        dimensions (list): Layer dimensions [input, hidden1, hidden2, output]
        learning_rate (float): Learning rate for optimization
        use_dropout (bool): Whether to apply dropout during training
        use_adam (bool): Whether to use Adam optimizer
    """
    
    def __init__(
        self,
        y,
        use_dropout=False,
        use_adam=False,
        lr=0.01,
        batch_size=64,
        dropout_prob=0.3,
    ):
        """
        Initialize the neural network.
        
        Args:
            y (np.ndarray): Ground truth labels for initialization
            use_dropout (bool): Enable dropout regularization
            use_adam (bool): Use Adam optimizer instead of vanilla SGD
            lr (float): Learning rate
            batch_size (int): Mini-batch size for training
            dropout_prob (float): Probability of dropping neurons (0-1)
        """
        self.y = y
        
        # Network architecture
        self.y_hat = np.zeros((self.y.shape[0], 3))
        self.dimensions = [8, 15, 7, 3]
        self.alpha = 0.05
        self.eps = 1e-8
        
        # Dropout configuration
        self.use_dropout = use_dropout
        self.dropout_prob = dropout_prob
        
        # Network state
        self.parameters = {}
        self.cache = {}
        self.loss = []
        self.batch_y = []
        
        # Training configuration
        self.iteration = 0
        self.batch_size = batch_size
        self.learning_rate = lr
        self.sample_count = self.y.shape[0]
        self._estimator_type = "regression"
        self.neural_net_type = "SiLU -> SiLU -> Softmax"
        
        # Adam optimizer parameters
        self.use_adam = use_adam
        self.t = 1
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.first_moment = {}
        self.second_moment = {}

    def init_parameters(self, param=None):
        """
        Initialize network weights and biases.
        
        Uses Xavier/He initialization for weights to prevent vanishing/exploding gradients.
        
        Args:
            param (dict, optional): Pre-defined parameters to use instead of random init
        """
        if param is None:
            np.random.seed(0)
            self.parameters["theta1"] = np.random.randn(
                self.dimensions[0], self.dimensions[1]
            ) / np.sqrt(self.dimensions[0])
            self.parameters["b1"] = np.zeros((self.dimensions[1]))
            self.parameters["theta2"] = np.random.randn(
                self.dimensions[1], self.dimensions[2]
            ) / np.sqrt(self.dimensions[1])
            self.parameters["b2"] = np.zeros((self.dimensions[2]))
            self.parameters["theta3"] = np.random.randn(
                self.dimensions[2], self.dimensions[3]
            ) / np.sqrt(self.dimensions[2])
            self.parameters["b3"] = np.zeros((self.dimensions[3]))
        else:
            self.parameters = param
            self.parameters["theta1"] = self.parameters["theta1"]
            self.parameters["theta2"] = self.parameters["theta2"]
            self.parameters["theta3"] = self.parameters["theta3"]
            self.parameters["b1"] = self.parameters["b1"]
            self.parameters["b2"] = self.parameters["b2"]
            self.parameters["b3"] = self.parameters["b3"]

        # Initialize Adam moment estimates
        for parameter in self.parameters:
            self.first_moment[parameter] = np.zeros_like(self.parameters[parameter])
            self.second_moment[parameter] = np.zeros_like(self.parameters[parameter])

    def softmax(self, u):
        """
        Compute softmax activation (numerically stable version).
        
        Converts logits to probability distribution over classes.
        Uses max subtraction for numerical stability to prevent overflow.
        
        Args:
            u (np.ndarray): Input logits of shape (N, D)
            
        Returns:
            np.ndarray: Probability distributions of shape (N, D)
        """
        u_stable = u - np.max(u, axis=1, keepdims=True)
        exp_u = np.exp(u_stable)
        return exp_u / np.sum(exp_u, axis=1, keepdims=True)

    def silu(self, u):
        """
        Compute SiLU (Sigmoid Linear Unit) activation, also known as Swish.
        
        SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
        
        Benefits over ReLU:
            - Smooth, non-monotonic function
            - Retains gradients for negative inputs
            - Self-gated activation
        
        Args:
            u (np.ndarray): Input array of any shape
            
        Returns:
            np.ndarray: Activated output, same shape as input
        """
        sigmoid = 1 / (1 + np.exp(-u))
        return u * sigmoid

    def derivative_silu(self, x):
        """
        Compute the derivative of SiLU activation.
        
        d/dx[SiLU(x)] = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        
        Args:
            x (np.ndarray): Input array
            
        Returns:
            np.ndarray: Derivative values
        """
        sigmoid = 1 / (1 + np.exp(-x))
        return sigmoid * (1 + x * (1 - sigmoid))

    @staticmethod
    def _dropout(u, prob):
        """
        Apply inverted dropout regularization.
        
        Randomly zeroes elements with probability `prob` during training.
        Scales remaining values by 1/(1-prob) to maintain expected values.
        
        Args:
            u (np.ndarray): Input tensor of shape (N, D)
            prob (float): Dropout probability (0-1)
            
        Returns:
            tuple: (output tensor, dropout mask)
        """
        if prob <= 0.0:
            mask = np.ones_like(u)
            return u, mask
        mask = (np.random.rand(*u.shape) > prob).astype(u.dtype)
        u_after_dropout = u * mask / (1.0 - prob)
        return u_after_dropout, mask

    def cross_entropy_loss(self, y, y_hat):
        """
        Compute categorical cross-entropy loss.
        
        L = -1/N * sum(y * log(y_hat))
        
        Args:
            y (np.ndarray): One-hot encoded ground truth labels (N, D)
            y_hat (np.ndarray): Predicted probabilities (N, D)
            
        Returns:
            float: Average cross-entropy loss
        """
        eps = 1e-15
        y_hat = np.clip(y_hat, eps, 1 - eps)
        loss = -np.sum(y * np.log(y_hat)) / y.shape[0]
        return loss

    def forward(self, x, use_dropout):
        """
        Perform forward propagation through the network.
        
        Architecture: X -> Linear -> SiLU -> [Dropout] -> Linear -> SiLU -> Linear -> Softmax
        
        Args:
            x (np.ndarray): Input features of shape (N, input_dim)
            use_dropout (bool): Whether to apply dropout
            
        Returns:
            np.ndarray: Output probabilities of shape (N, num_classes)
        """
        self.cache["X"] = x
        
        # Layer 1: Linear + SiLU + Dropout
        u1 = np.dot(x, self.parameters["theta1"]) + self.parameters["b1"]
        o1 = self.silu(u1)
        if use_dropout:
            o1, dropout_mask = self._dropout(o1, self.dropout_prob)
            self.cache["mask"] = dropout_mask
        self.cache["u1"], self.cache["o1"] = u1, o1
        
        # Layer 2: Linear + SiLU
        u2 = np.dot(o1, self.parameters["theta2"]) + self.parameters["b2"]
        o2 = self.silu(u2)
        self.cache["u2"], self.cache["o2"] = u2, o2
        
        # Output layer: Linear + Softmax
        u3 = np.dot(o2, self.parameters["theta3"]) + self.parameters["b3"]
        o3 = self.softmax(u3)
        self.cache["u3"], self.cache["o3"] = u3, o3
        
        return o3

    def compute_gradients(self, y, yh):
        """
        Compute gradients via backpropagation.
        
        Uses chain rule to compute gradients of loss w.r.t. all parameters.
        Handles dropout mask if dropout was used during forward pass.
        
        Args:
            y (np.ndarray): Ground truth labels (N, D)
            yh (np.ndarray): Predicted outputs (N, D)
            
        Returns:
            dict: Gradients for all parameters
        """
        X = self.cache["X"]
        u1 = self.cache["u1"]
        o1 = self.cache["o1"]
        u2 = self.cache["u2"]
        o2 = self.cache["o2"]
        
        # Output layer gradients
        dLoss_u3 = yh - y
        dLoss_theta3 = o2.T @ dLoss_u3 / y.shape[0]
        dLoss_b3 = np.sum(dLoss_u3, axis=0) / y.shape[0]
        
        # Second hidden layer gradients
        dLoss_o2 = dLoss_u3 @ self.parameters["theta3"].T
        dLoss_u2 = dLoss_o2 * self.derivative_silu(u2)
        dLoss_theta2 = o1.T @ dLoss_u2 / y.shape[0]
        dLoss_b2 = np.sum(dLoss_u2, axis=0) / y.shape[0]
        
        # First hidden layer gradients (with dropout handling)
        dLoss_o1 = dLoss_u2 @ self.parameters["theta2"].T
        if self.use_dropout and "mask" in self.cache:
            mask = self.cache["mask"]
            dLoss_o1 = dLoss_o1 * mask / (1.0 - self.dropout_prob)
        dLoss_u1 = dLoss_o1 * self.derivative_silu(u1)
        dLoss_theta1 = X.T @ dLoss_u1 / y.shape[0]
        dLoss_b1 = np.sum(dLoss_u1, axis=0) / y.shape[0]
        
        return {
            "theta1": dLoss_theta1,
            "b1": dLoss_b1,
            "theta2": dLoss_theta2,
            "b2": dLoss_b2,
            "theta3": dLoss_theta3,
            "b3": dLoss_b3,
        }

    def update_weights(self, dLoss):
        """
        Update network parameters using gradients.
        
        Supports both vanilla SGD and Adam optimizer.
        Adam uses adaptive learning rates with momentum.
        
        Args:
            dLoss (dict): Gradients for all parameters
        """
        for param in ["theta1", "b1", "theta2", "b2", "theta3", "b3"]:
            grad = dLoss[param]
            if self.use_adam:
                # Adam optimizer update
                self.first_moment[param] = self.beta1 * self.first_moment[param] + (1 - self.beta1) * grad
                self.second_moment[param] = self.beta2 * self.second_moment[param] + (1 - self.beta2) * (grad ** 2)
                m_hat = self.first_moment[param] / (1 - self.beta1 ** self.t)
                v_hat = self.second_moment[param] / (1 - self.beta2 ** self.t)
                self.parameters[param] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)
            else:
                # Vanilla SGD update
                self.parameters[param] -= self.learning_rate * grad
        
        if self.use_adam:
            self.t += 1

    def gradient_descent(self, x, y, iter=60000, local_test=False):
        """
        Train using full-batch gradient descent.
        
        Args:
            x (np.ndarray): Training features
            y (np.ndarray): Training labels
            iter (int): Number of iterations
            local_test (bool): If True, print every iteration
        """
        self.init_parameters()
        
        for i in range(iter):
            yh = self.forward(x, self.use_dropout)
            loss = self.cross_entropy_loss(y, yh)
            self.backward(y, yh)
            
            print_multiple = 1 if local_test else 1000
            if i % print_multiple == 0:
                print(f"Loss after iteration {i}: {loss}")
                self.loss.append(loss)
                self.batch_y.append(y)
            
            if not self.use_adam:
                self.t += 1

    def minibatch_gradient_descent(self, x, y, iter=60000, local_test=False):
        """
        Train using mini-batch gradient descent.
        
        Processes data in batches with wraparound for incomplete batches.
        
        Args:
            x (np.ndarray): Training features
            y (np.ndarray): Training labels  
            iter (int): Number of iterations
            local_test (bool): If True, print every iteration
        """
        self.init_parameters()
        n_samples = x.shape[0]
        batch_size = self.batch_size
        
        for i in range(iter):
            start = (i * batch_size) % n_samples
            end = start + batch_size
            
            if end <= n_samples:
                x_batch = x[start:end]
                y_batch = y[start:end]
            else:
                # Wraparound for incomplete batches
                idx = np.arange(start, end) % n_samples
                x_batch = x[idx]
                y_batch = y[idx]
            
            yh = self.forward(x_batch, self.use_dropout)
            loss = self.cross_entropy_loss(y_batch, yh)
            self.backward(y_batch, yh)
            
            print_multiple = 1 if local_test else 1000
            if i % print_multiple == 0:
                print(f"Loss after iteration {i}: {loss}")
                self.loss.append(loss)
                self.batch_y.append(y_batch)
            
            if not self.use_adam:
                self.t += 1

    def backward(self, y, yh):
        """
        Perform backward pass: compute gradients and update weights.
        
        Args:
            y (np.ndarray): Ground truth labels
            yh (np.ndarray): Predicted outputs
        """
        grads = self.compute_gradients(y, yh)
        self.update_weights(grads)

    def predict(self, x):
        """
        Make predictions on new data.
        
        Args:
            x (np.ndarray): Input features of shape (N, input_dim)
            
        Returns:
            np.ndarray: Predicted class labels of shape (N,)
        """
        yh = self.forward(x, False)
        pred = np.argmax(yh, axis=1)
        return pred
