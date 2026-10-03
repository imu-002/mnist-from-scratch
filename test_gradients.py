"""Check the hand-written backward pass against numerical gradients.

    python test_gradients.py

For every parameter we nudge it by a tiny amount, measure how the loss
changes, and compare that slope with what backward() computed. If the
backprop maths is wrong anywhere, the two numbers will disagree.
"""

import numpy as np

from network import Network, cross_entropy, softmax


def numerical_grad(net, x, y, param, eps=1e-5):
    grad = np.zeros_like(param)
    it = np.nditer(param, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        old = param[i]

        param[i] = old + eps
        loss_plus, _ = cross_entropy(net.forward(x), y)
        param[i] = old - eps
        loss_minus, _ = cross_entropy(net.forward(x), y)
        param[i] = old

        grad[i] = (loss_plus - loss_minus) / (2 * eps)
        it.iternext()
    return grad


def test_backward_matches_numerical():
    rng = np.random.default_rng(1)
    net = Network([12, 9, 7, 4], seed=1)
    x = rng.normal(size=(5, 12))
    y = rng.integers(0, 4, size=5)

    loss, grad = cross_entropy(net.forward(x), y)
    net.backward(grad)

    for param, analytic in net.params():
        analytic = analytic.copy()
        numeric = numerical_grad(net, x, y, param)
        diff = np.abs(analytic - numeric).max()
        scale = max(np.abs(analytic).max(), np.abs(numeric).max(), 1e-12)
        assert diff / scale < 1e-6, f"gradient mismatch: {diff / scale:.2e}"


def test_softmax_rows_sum_to_one():
    logits = np.array([[1000.0, 1001.0, 999.0], [-5.0, 0.0, 5.0]])
    probs = softmax(logits)
    assert np.all(np.isfinite(probs))  # large logits must not overflow
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_network_can_overfit_tiny_batch():
    # If the training loop is correct the net should memorise 20 points.
    from network import SGD

    rng = np.random.default_rng(2)
    x = rng.normal(size=(20, 10))
    y = rng.integers(0, 3, size=20)
    net = Network([10, 32, 3], seed=2)
    opt = SGD(net.params, lr=0.1, momentum=0.9)
    for _ in range(300):
        loss, grad = cross_entropy(net.forward(x), y)
        net.backward(grad)
        opt.step()
    assert loss < 0.01
    assert (net.predict(x) == y).all()


if __name__ == "__main__":
    test_backward_matches_numerical()
    print("backward pass matches numerical gradients")
    test_softmax_rows_sum_to_one()
    print("softmax is stable")
    test_network_can_overfit_tiny_batch()
    print("network overfits a tiny batch")
    print("all checks passed")
