"""Evaluate the saved network and draw the figures used in the README.

    python evaluate.py
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np

import data
from network import Network, softmax

BLUE = "#2a78d6"
ORANGE = "#d9622b"
GREY = "#52514e"


def confusion_matrix(y_true, y_pred):
    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def plot_curves(history, path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(epochs, history["train_loss"], color=BLUE, lw=2, label="training")
    ax.plot(epochs, history["val_loss"], color=ORANGE, lw=2, label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("Loss during training")
    ax.set_ylim(bottom=0)
    ax.set_xticks(range(2, len(epochs) + 1, 2))
    ax.grid(axis="y", color="#e6e5e0", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(cm, path):
    # Blank the diagonal so the mistakes are visible. Correct counts are in
    # the hundreds and would wash everything else out.
    errors = cm.copy()
    np.fill_diagonal(errors, 0)

    fig, ax = plt.subplots(figsize=(6, 5.4))
    im = ax.imshow(errors, cmap="Blues")
    for i in range(10):
        for j in range(10):
            if i != j and errors[i, j] > 0:
                dark = errors[i, j] > errors.max() * 0.6
                ax.text(j, i, errors[i, j], ha="center", va="center",
                        fontsize=8, color="white" if dark else "#0b0b0b")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("predicted digit")
    ax.set_ylabel("true digit")
    ax.set_title("Test set mistakes (correct predictions hidden)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="number of images")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_mistakes(x, y_true, y_pred, confidence, path, n=16):
    wrong = np.where(y_true != y_pred)[0]
    # most confident mistakes first, they are the interesting ones
    wrong = wrong[np.argsort(-confidence[wrong])][:n]

    fig, axes = plt.subplots(2, n // 2, figsize=(n // 2 * 1.3, 3.4))
    for ax, i in zip(axes.ravel(), wrong):
        ax.imshow(x[i].reshape(28, 28), cmap="gray_r")
        ax.set_title(f"{y_true[i]} read as {y_pred[i]}", fontsize=8, color=GREY)
        ax.axis("off")
    fig.suptitle("Most confident mistakes on the test set", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    with open(os.path.join(args.out, "history.json")) as f:
        history = json.load(f)

    _, _, x_test, y_test = data.load()
    net = Network([784] + history["args"]["hidden"] + [10])
    net.load(os.path.join(args.out, "weights.npz"))

    probs = softmax(net.forward(x_test))
    y_pred = probs.argmax(axis=1)
    confidence = probs.max(axis=1)

    cm = confusion_matrix(y_test, y_pred)
    print(f"test accuracy: {(y_pred == y_test).mean():.4f} "
          f"({(y_pred != y_test).sum()} wrong out of {len(y_test)})\n")

    print("digit  accuracy  most often mistaken for")
    for d in range(10):
        row = cm[d].copy()
        acc = row[d] / row.sum()
        row[d] = 0
        print(f"  {d}     {acc:.4f}    {row.argmax()} ({row.max()} times)")

    plot_curves(history, os.path.join(args.out, "loss_curve.png"))
    plot_confusion(cm, os.path.join(args.out, "confusion_matrix.png"))
    plot_mistakes(x_test, y_test, y_pred, confidence,
                  os.path.join(args.out, "mistakes.png"))
    print(f"\nfigures written to {args.out}/")


if __name__ == "__main__":
    main()
