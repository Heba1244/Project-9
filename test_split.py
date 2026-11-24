"""Checks for data_and_split.py on small fake tables. Run: pytest -v test_split.py"""
import numpy as np
import pandas as pd
import pytest
from data_and_split import load_model_table, make_test_split, grouped_folds


@pytest.fixture()
def pack(tmp_path):
    rng = np.random.default_rng(0)
    rows, i = [], 0
    for f in range(12):                                   # 12 families x 5 cell types x 4 neurons
        for t in range(5):
            for _ in range(4):
                rows.append((i, f"F{f}", f"F{f}_t{t}", False)); i += 1
    rows += [(10_000 + k, "tiny", "tiny_t0", False) for k in range(3)]               # too small: dropped
    rows += [(11_000 + k, "lonely", "lonely_t0", False) for k in range(15)]           # one cell type: dropped
    rows += [(12_000 + k, "putative_primary", f"p{k}", False) for k in range(30)]     # left-out label
    rows += [(13_000 + k, "F0", f"opt{k}", True) for k in range(10)]                  # optic: dropped
    rows += [(14_000 + k, "F1", None, False) for k in range(4)]                       # no cell_type: dropped
    n = pd.DataFrame(rows, columns=["neuron_id", "hemilineage", "cell_type", "is_optic"])
    f = pd.DataFrame({"neuron_id": n.neuron_id, "frac_in_A": rng.random(len(n)), "frac_out_A": rng.random(len(n)),
                      "total_in_syn": 1, "total_out_syn": 1, "in_degree": 1, "out_degree": 1})
    n.to_parquet(tmp_path / "neurons_fafb.parquet"); f.to_parquet(tmp_path / "features_fafb.parquet")
    return tmp_path


def test_filters(pack):
    table, share, raw = load_model_table(pack)
    assert set(table.hemilineage) == {f"F{i}" for i in range(12)}
    assert table.cell_type.notna().all()
    assert share == ["frac_in_A", "frac_out_A"]
    assert len(table) == 240                                                       # only the 12 clean families remain


def test_test_split_keeps_cell_types_whole_and_every_family_in_training(pack):
    table, _, _ = load_model_table(pack)
    dev, test = make_test_split(table, 0.2, seed=42)
    assert set(dev.cell_type).isdisjoint(set(test.cell_type))
    assert set(dev.hemilineage) == set(table.hemilineage)
    assert 0.1 < len(test) / len(table) < 0.3
    dev2, test2 = make_test_split(table, 0.2, seed=42)           # same seed, same split
    assert set(test.cell_type) == set(test2.cell_type)


def test_folds_are_grouped_and_cover_everything(pack):
    table, _, _ = load_model_table(pack)
    dev, _ = make_test_split(table)
    seen = {}
    for r, f, tr, va in grouped_folds(dev, n_splits=5, n_repeats=5):
        assert set(dev.cell_type[tr]).isdisjoint(set(dev.cell_type[va]))
        assert len(tr) + len(va) == len(dev)
        seen.setdefault(r, []).extend(va.tolist())
    assert all(sorted(v) == list(range(len(dev))) for v in seen.values())     # each neuron validated once per repeat
    assert len(seen) == 5


def test_repeats_use_different_shuffles(pack):
    table, _, _ = load_model_table(pack)
    dev, _ = make_test_split(table)
    first = {}
    for r, f, tr, va in grouped_folds(dev):
        if f == 0:
            first[r] = frozenset(va.tolist())
    assert len(set(first.values())) > 1
