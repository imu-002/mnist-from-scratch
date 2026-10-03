"""Download MNIST and load it into NumPy arrays."""

import gzip
import hashlib
import os
import urllib.request

import numpy as np

# The original site (yann.lecun.com) no longer serves the files reliably,
# so these are the two mirrors most libraries use now.
MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]

# file name -> md5 of the .gz file
FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


def _md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def download(data_dir="data", mirrors=MIRRORS):
    """Fetch the four MNIST files into data_dir, skipping ones already there."""
    os.makedirs(data_dir, exist_ok=True)
    for name, checksum in FILES.items():
        path = os.path.join(data_dir, name)
        if os.path.exists(path) and _md5(path) == checksum:
            continue

        for base in mirrors:
            try:
                print(f"downloading {name} from {base}")
                urllib.request.urlretrieve(base + name, path)
            except OSError as err:
                print(f"  failed: {err}")
                continue
            if _md5(path) == checksum:
                break
            print("  checksum did not match, trying next mirror")
        else:
            raise RuntimeError(f"could not download {name}")


def _read_images(path):
    # idx3 format: 16 byte header (magic, count, rows, cols), then raw pixels
    with gzip.open(path, "rb") as f:
        raw = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    return raw.reshape(-1, 28 * 28)


def _read_labels(path):
    # idx1 format: 8 byte header (magic, count), then one byte per label
    with gzip.open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.uint8, offset=8)


def load(data_dir="data"):
    """Return (x_train, y_train, x_test, y_test).

    Images are float32 in [0, 1] with shape (n, 784). Labels are ints 0-9.
    """
    download(data_dir)

    def p(name):
        return os.path.join(data_dir, name)

    x_train = _read_images(p("train-images-idx3-ubyte.gz")).astype(np.float32) / 255.0
    y_train = _read_labels(p("train-labels-idx1-ubyte.gz")).astype(np.int64)
    x_test = _read_images(p("t10k-images-idx3-ubyte.gz")).astype(np.float32) / 255.0
    y_test = _read_labels(p("t10k-labels-idx1-ubyte.gz")).astype(np.int64)
    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load()
    print("train:", x_train.shape, y_train.shape)
    print("test: ", x_test.shape, y_test.shape)
