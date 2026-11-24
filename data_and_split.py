"""Project 9: data loading and splitting (no model is trained here).

load_model_table  : central-brain neurons with a usable hemilineage + their share features
make_test_split   : hold out about 20% of cell types as the test set (used once, at the end)
grouped_folds     : 5-fold cross-validation grouped by cell_type, repeated over several shuffles
"""
from pathlib import Path
import numpy as np
import pandas as pd

LEFT_OUT_KEYS = ("prim", "putative", "missing", "unknown")  # primary, putative_primary, __prim, missing, unknown
MIN_NEURONS = 10
MIN_CELL_TYPES = 2
SEED = 42


def load_model_table(pack="data/course", min_neurons=MIN_NEURONS, min_cell_types=MIN_CELL_TYPES):
    """Rows = central-brain neurons of kept hemilineage families. Columns = neuron_id, hemilineage, cell_type,
    the share features (frac_in_*, frac_out_*) and the raw-count comparison set."""
    pack = Path(pack)
    neurons = pd.read_parquet(pack / "neurons_fafb.parquet")
    feats = pd.read_parquet(pack / "features_fafb.parquet")

    c = neurons[~neurons["is_optic"]].dropna(subset=["hemilineage"])             # central brain, labelled
    lab = c["hemilineage"].astype(str).str.lower()
    c = c[~lab.apply(lambda v: any(k in v for k in LEFT_OUT_KEYS))]               # leave out primary/putative/etc.

    fam = c.groupby("hemilineage").agg(n=("neuron_id", "size"), t=("cell_type", "nunique"))
    keep = fam[(fam.n >= min_neurons) & (fam.t >= min_cell_types)].index
    c = c[c["hemilineage"].isin(keep)]
    c = c.dropna(subset=["cell_type"])                                            # cannot be grouped without a cell type

    share_cols = [x for x in feats.columns if x.startswith(("frac_in_", "frac_out_"))]
    raw_cols = ["total_in_syn", "total_out_syn", "in_degree", "out_degree"]
    table = c[["neuron_id", "hemilineage", "cell_type"]].merge(
        feats[["neuron_id"] + share_cols + raw_cols], on="neuron_id", how="left")
    return table.reset_index(drop=True), share_cols, raw_cols


def make_test_split(table, test_frac=0.2, seed=SEED):
    """Hold out whole cell types. Every family keeps at least one cell type in training.
    Returns (dev_table, test_table). Call once, save the test cell types, and do not touch the test table until the end."""
    rng = np.random.default_rng(seed)
    types = table.groupby("cell_type")["hemilineage"].agg(lambda s: s.mode().iloc[0])   # family of each cell type
    n_target = int(round(test_frac * len(types)))
    train_left = types.value_counts().to_dict()                                   # cell types per family still in training
    test_types = []
    for ct in rng.permutation(types.index.to_numpy()):
        if len(test_types) >= n_target:
            break
        fam = types[ct]
        if train_left[fam] > 1:
            test_types.append(ct)
            train_left[fam] -= 1
    is_test = table["cell_type"].isin(test_types)
    return table[~is_test].reset_index(drop=True), table[is_test].reset_index(drop=True)


def grouped_folds(dev_table, n_splits=5, n_repeats=5, seed=SEED):
    """Yield (repeat, fold, train_idx, val_idx). Whole cell types stay on one side; each repeat reshuffles them."""
    types = dev_table["cell_type"].to_numpy()
    uniq = np.unique(types)
    for r in range(n_repeats):
        rng = np.random.default_rng(seed + r)
        fold_of = {ct: i % n_splits for i, ct in enumerate(rng.permutation(uniq))}
        folds = np.array([fold_of[ct] for ct in types])
        for f in range(n_splits):
            yield r, f, np.where(folds != f)[0], np.where(folds == f)[0]
