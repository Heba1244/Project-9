"""Decision tree classifier in pure NumPy (entropy criterion, axis-aligned splits)."""
import numpy as np


def entropy_from_counts(counts):
    """Entropy (base 2) of class counts; works on a 1D array or a 2D array of rows."""
    counts = np.asarray(counts, dtype=float)
    total = counts.sum(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(total > 0, counts / total, 0.0)
        logp = np.where(p > 0, np.log2(p), 0.0)
    return -(p * logp).sum(axis=-1)


class NumpyTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        self.tree_ = self._grow(X, y_idx, depth=0)
        return self

    def _best_split(self, X, y):
        n, d = X.shape
        onehot = np.eye(self.n_classes_)[y]
        parent = entropy_from_counts(onehot.sum(axis=0))
        best = (0.0, None, None)  # gain, feature, threshold
        for j in range(d):
            order = np.argsort(X[:, j], kind="stable")
            xs = X[order, j]
            left = np.cumsum(onehot[order], axis=0)[:-1]          # counts left of each cut
            right = onehot.sum(axis=0) - left
            n_left = np.arange(1, n)
            n_right = n - n_left
            child = (n_left * entropy_from_counts(left) + n_right * entropy_from_counts(right)) / n
            valid = xs[1:] > xs[:-1]                              # cut only between different values
            if not valid.any():
                continue
            gain = np.where(valid, parent - child, -np.inf)
            k = int(np.argmax(gain))
            if gain[k] > best[0] + 1e-12:
                best = (float(gain[k]), j, float((xs[k] + xs[k + 1]) / 2))
        return best

    def _grow(self, X, y, depth):
        counts = np.bincount(y, minlength=self.n_classes_)
        leaf = {"leaf": True, "pred": int(np.argmax(counts))}
        if len(y) < 2 or (counts > 0).sum() == 1:
            return leaf
        if self.max_depth is not None and depth >= self.max_depth:
            return leaf
        gain, j, thr = self._best_split(X, y)
        if j is None:
            return leaf
        mask = X[:, j] <= thr
        return {"leaf": False, "feature": j, "thr": thr,
                "left": self._grow(X[mask], y[mask], depth + 1),
                "right": self._grow(X[~mask], y[~mask], depth + 1)}

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        out = np.empty(len(X), dtype=int)
        for i, row in enumerate(X):
            node = self.tree_
            while not node["leaf"]:
                node = node["left"] if row[node["feature"]] <= node["thr"] else node["right"]
            out[i] = node["pred"]
        return self.classes_[out]

    def depth(self):
        def d(n): return 0 if n["leaf"] else 1 + max(d(n["left"]), d(n["right"]))
        return d(self.tree_)
