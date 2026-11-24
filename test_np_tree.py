"""Code checks for the NumPy tree.
Check 1: synthetic data with a known answer.  Check 2: same predictions as scikit-learn (entropy) at the same depth.
Run:  pytest -v test_np_tree.py
"""
import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier
from np_tree import NumpyTree, entropy_from_counts


# ---------- Check 1: synthetic data with a known answer ----------
def test_entropy_values():
    assert entropy_from_counts([5, 0]) == pytest.approx(0.0)
    assert entropy_from_counts([5, 5]) == pytest.approx(1.0)
    assert entropy_from_counts([1, 1, 1, 1]) == pytest.approx(2.0)


def test_single_rule_is_found_exactly():
    rng = np.random.default_rng(0)
    X = rng.random((300, 4))
    y = (X[:, 2] > 0.4).astype(int)                 # known answer: feature 2, threshold 0.4
    t = NumpyTree(max_depth=1).fit(X, y)
    assert t.tree_["feature"] == 2
    assert 0.35 < t.tree_["thr"] < 0.45
    assert (t.predict(X) == y).all()


def test_two_rule_three_class_problem():
    rng = np.random.default_rng(1)
    X = rng.random((600, 3))
    y = np.where(X[:, 0] < 0.3, 0, np.where(X[:, 1] > 0.6, 1, 2))   # known answer
    t = NumpyTree(max_depth=2).fit(X, y)
    assert t.tree_["feature"] in (0, 1)
    assert (t.predict(X) == y).all()
    Xn = rng.random((500, 3))
    yn = np.where(Xn[:, 0] < 0.3, 0, np.where(Xn[:, 1] > 0.6, 1, 2))
    assert (t.predict(Xn) == yn).mean() > 0.97       # generalises to fresh points


def test_pure_data_gives_one_leaf():
    X = np.random.default_rng(2).random((20, 3))
    t = NumpyTree().fit(X, np.zeros(20, dtype=int))
    assert t.tree_["leaf"] and t.depth() == 0


@pytest.mark.parametrize("d", [1, 2, 4])
def test_max_depth_is_respected(d):
    rng = np.random.default_rng(3)
    X = rng.random((400, 5)); y = rng.integers(0, 4, 400)
    assert NumpyTree(max_depth=d).fit(X, y).depth() <= d


# ---------- Check 2: same predictions as scikit-learn at the same depth ----------
def make_data(seed, n=500, n_new=300):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(5, 6)) * 1.5
    y = rng.integers(0, 5, n)
    X = centers[y] + rng.normal(size=(n, 6))          # noisy, overlapping classes
    Xnew = centers[rng.integers(0, 5, n_new)] + rng.normal(size=(n_new, 6))
    return X, y, Xnew


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("seed", range(10))
def test_identical_predictions_at_shallow_depth(depth, seed):
    """Where no two splits tie, both trees must make exactly the same predictions."""
    X, y, Xnew = make_data(seed)
    mine = NumpyTree(max_depth=depth).fit(X, y)
    ref = DecisionTreeClassifier(criterion="entropy", max_depth=depth, random_state=0).fit(X, y)
    assert (mine.predict(X) == ref.predict(X)).all()
    assert (mine.predict(Xnew) == ref.predict(Xnew)).all()


@pytest.mark.parametrize("depth", [5, 8, None])
@pytest.mark.parametrize("seed", range(5))
def test_close_agreement_at_deeper_depth(depth, seed):
    """Deep in the tree, small nodes can have equal-gain splits, and scikit-learn breaks those ties
    with a random feature order, so exact equality is not guaranteed. Same root split, near-identical output."""
    X, y, Xnew = make_data(seed)
    mine = NumpyTree(max_depth=depth).fit(X, y)
    ref = DecisionTreeClassifier(criterion="entropy", max_depth=depth, random_state=0).fit(X, y)
    assert mine.tree_["feature"] == ref.tree_.feature[0]
    assert mine.tree_["thr"] == pytest.approx(ref.tree_.threshold[0], abs=1e-4)
    assert (mine.predict(X) == ref.predict(X)).mean() > 0.99
    assert (mine.predict(Xnew) == ref.predict(Xnew)).mean() > 0.90
    if depth is None:                                  # both grow to pure leaves
        assert (mine.predict(X) == y).all() and (ref.predict(X) == y).all()
