"""Train the network on MNIST.

    python train.py
    python train.py --epochs 30 --hidden 256 128
"""

import argparse
import json
import os
import time

import numpy as np

import data
from network import SGD, Network, cross_entropy


def accuracy(net, x, y, batch_size=1000):
    correct = 0
    for i in range(0, len(x), batch_size):
        correct += (net.predict(x[i:i + batch_size]) == y[i:i + batch_size]).sum()
    return correct / len(x)


def mean_loss(net, x, y, batch_size=1000):
    total = 0.0
    for i in range(0, len(x), batch_size):
        loss, _ = cross_entropy(net.forward(x[i:i + batch_size]), y[i:i + batch_size])
        total += loss * len(x[i:i + batch_size])
    return total / len(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, nargs="*", default=[128, 64])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    x_train, y_train, x_test, y_test = data.load()

    # Hold out the last 5,000 training images for validation. The test set
    # is only touched once, at the very end.
    x_val, y_val = x_train[-5000:], y_train[-5000:]
    x_train, y_train = x_train[:-5000], y_train[:-5000]

    net = Network([784] + args.hidden + [10], seed=args.seed)
    opt = SGD(net.params, lr=args.lr, momentum=args.momentum,
              weight_decay=args.weight_decay)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val = 0.0
    weights_path = os.path.join(args.out, "weights.npz")
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        # halve the learning rate every 8 epochs
        opt.lr = args.lr * (0.5 ** ((epoch - 1) // 8))

        order = rng.permutation(len(x_train))
        running = 0.0
        for i in range(0, len(order), args.batch_size):
            idx = order[i:i + args.batch_size]
            loss, grad = cross_entropy(net.forward(x_train[idx]), y_train[idx])
            net.backward(grad)
            opt.step()
            running += loss * len(idx)

        train_loss = running / len(x_train)
        val_loss = mean_loss(net, x_val, y_val)
        val_acc = accuracy(net, x_val, y_val)
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))

        note = ""
        if val_acc > best_val:
            best_val = val_acc
            net.save(weights_path)
            note = "  (saved)"
        print(f"epoch {epoch:2d}  train loss {train_loss:.4f}  "
              f"val loss {val_loss:.4f}  val acc {val_acc:.4f}{note}")

    elapsed = time.time() - start

    # Evaluate the checkpoint that did best on validation.
    net.load(weights_path)
    test_acc = accuracy(net, x_test, y_test)
    print(f"\nbest val acc {best_val:.4f}")
    print(f"test acc     {test_acc:.4f}")
    print(f"training took {elapsed:.0f}s")

    history["best_val_acc"] = float(best_val)
    history["test_acc"] = float(test_acc)
    history["train_seconds"] = round(elapsed, 1)
    history["args"] = vars(args)
    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
