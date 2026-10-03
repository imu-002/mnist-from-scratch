"""A small fully connected neural network in plain NumPy.

Nothing here uses an autograd library. Every layer computes its own
gradient in backward(), and the network just chains them together.
"""

import numpy as np


class Dense:
    """Fully connected layer: out = x @ W + b"""

    def __init__(self, n_in, n_out, rng):
        # He initialisation. Keeps the signal from shrinking or blowing up
        # as it passes through ReLU layers.
        self.W = rng.normal(0.0, np.sqrt(2.0 / n_in), size=(n_in, n_out))
        self.b = np.zeros(n_out)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x  # needed again in backward
        return x @ self.W + self.b

    def backward(self, grad_out):
        # grad_out is dLoss/d(out), shape (batch, n_out)
        self.dW = self.x.T @ grad_out
        self.db = grad_out.sum(axis=0)
        return grad_out @ self.W.T  # dLoss/dx, passed to the layer before

    def params(self):
        return [(self.W, self.dW), (self.b, self.db)]


class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad_out):
        return grad_out * self.mask

    def params(self):
        return []


def softmax(logits):
    # subtract the row max first so exp() cannot overflow
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def cross_entropy(logits, labels):
    """Mean cross-entropy loss and its gradient with respect to the logits.

    Softmax and cross-entropy are done together because the combined
    gradient is just (probabilities - one_hot) / batch_size.
    """
    n = logits.shape[0]
    probs = softmax(logits)
    loss = -np.log(probs[np.arange(n), labels] + 1e-12).mean()

    grad = probs.copy()
    grad[np.arange(n), labels] -= 1.0
    grad /= n
    return loss, grad


class Network:
    def __init__(self, sizes, seed=0):
        """sizes like [784, 128, 64, 10] gives two hidden layers."""
        rng = np.random.default_rng(seed)
        self.layers = []
        for i in range(len(sizes) - 1):
            self.layers.append(Dense(sizes[i], sizes[i + 1], rng))
            if i < len(sizes) - 2:  # no activation after the output layer
                self.layers.append(ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x  # logits

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def params(self):
        out = []
        for layer in self.layers:
            out.extend(layer.params())
        return out

    def predict(self, x):
        return self.forward(x).argmax(axis=1)

    def save(self, path):
        arrays = {}
        dense = [l for l in self.layers if isinstance(l, Dense)]
        for i, layer in enumerate(dense):
            arrays[f"W{i}"] = layer.W
            arrays[f"b{i}"] = layer.b
        np.savez(path, **arrays)

    def load(self, path):
        arrays = np.load(path)
        dense = [l for l in self.layers if isinstance(l, Dense)]
        for i, layer in enumerate(dense):
            layer.W[...] = arrays[f"W{i}"]
            layer.b[...] = arrays[f"b{i}"]


class SGD:
    """Stochastic gradient descent with momentum and L2 weight decay."""

    def __init__(self, params_fn, lr=0.05, momentum=0.9, weight_decay=0.0):
        self.params_fn = params_fn
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = None

    def step(self):
        params = self.params_fn()
        if self.velocity is None:
            self.velocity = [np.zeros_like(p) for p, _ in params]

        for (p, grad), v in zip(params, self.velocity):
            if self.weight_decay and p.ndim == 2:  # decay weights, not biases
                grad = grad + self.weight_decay * p
            v *= self.momentum
            v -= self.lr * grad
            p += v
