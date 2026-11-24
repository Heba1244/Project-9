"""
Project 9 first look (no model is run here). Prints every number the lock note needs.
Usage:  python first_look.py [path/to/data/course]
Outputs: first_look_numbers.md and figures fig1_family_sizes.png, fig2_raw_counts.png
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import skew

PACK = Path(sys.argv[1] if len(sys.argv) > 1 else "data/course")
out = []
def say(key, value, note=""):
    line = f"{key} = {value}" + (f"   ({note})" if note else "")
    print(line); out.append(line)

neurons = pd.read_parquet(PACK / "neurons_fafb.parquet")
feats = pd.read_parquet(PACK / "features_fafb.parquet")
edges = pd.read_parquet(PACK / "edges_fafb.parquet")
assert neurons["neuron_id"].dtype == "int64"

# ---- Data: population and label coverage
say("N_total", len(neurons), "course reference: 139,255")
cen, opt = neurons[~neurons["is_optic"]], neurons[neurons["is_optic"]]
say("N_central", len(cen)); say("N_optic", len(opt))
say("P (% optic neurons with a hemilineage)", round(100 * opt["hemilineage"].notna().mean(), 2))
say("Q (% central neurons with a hemilineage)", round(100 * cen["hemilineage"].notna().mean(), 2))

# ---- Threshold: how many connections survive at 5, 10, 31
tot = len(edges)
for t, key in [(5, "R"), (10, "S_default"), (31, "U")]:
    n = int((edges["syn_count"] >= t).sum())
    say(f"{key} (edges >= {t} synapses)", n, f"{100*n/tot:.1f}% of {tot:,} edges")

# ---- Target: hemilineage families in the central brain, minus labels left out
LEFT_OUT_KEYS = ("prim", "putative", "missing", "unknown")   # primary / putative_primary / missing / unknown
c = cen.dropna(subset=["hemilineage"])
lab = c["hemilineage"].astype(str).str.lower()
is_left_out = lab.apply(lambda v: any(k in v for k in LEFT_OUT_KEYS))
print("\nLabels left out (check this list is what you want to drop):")
print(c.loc[is_left_out, "hemilineage"].value_counts().head(25).to_string())
say("neurons dropped as left-out labels", int(is_left_out.sum()), f"{c.loc[is_left_out,'hemilineage'].nunique()} distinct labels")
c = c[~is_left_out]
fam = c.groupby("hemilineage").agg(n_neurons=("neuron_id", "size"),
                                   n_cell_types=("cell_type", "nunique"))
fam.sort_values("n_neurons", ascending=False).to_csv("table_families.csv")
say("K_total (families with a usable label)", len(fam))
print("\nFamilies kept for each candidate cutoff (neurons >= N and cell types >= 2):")
for N in (3, 5, 10, 20, 30):
    keep = fam[(fam.n_neurons >= N) & (fam.n_cell_types >= 2)]
    say(f"  N={N}: K", len(keep), f"neurons M={int(keep.n_neurons.sum())}, cell types C={int(keep.n_cell_types.sum())}")
only1 = int((fam.n_cell_types < 2).sum())
say("families with only 1 cell type (would be unpredictable)", only1)

fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(fam["n_neurons"], bins=40, log=True); ax.set_xlabel("neurons per hemilineage"); ax.set_ylabel("families (log)")
fig.tight_layout(); fig.savefig("fig1_family_sizes.png", dpi=150)

# ---- Features: skew of raw counts
for col in ["total_in_syn", "total_out_syn", "in_degree", "out_degree"]:
    v = feats[col].to_numpy(dtype=float)
    say(f"{col}: median / max / skew", f"{np.median(v):.0f} / {v.max():.0f} / {skew(v):.2f}")
fig, axs = plt.subplots(1, 2, figsize=(9, 3.5))
axs[0].hist(feats["total_in_syn"], bins=60, log=True); axs[0].set_title("total_in_syn (raw)")
axs[1].hist(feats["total_out_syn"], bins=60, log=True); axs[1].set_title("total_out_syn (raw)")
fig.tight_layout(); fig.savefig("fig2_raw_counts.png", dpi=150)

# ---- Missing values: neurons with no inputs / no outputs (all edges, and at threshold 10)
no_in = feats["total_in_syn"] == 0; no_out = feats["total_out_syn"] == 0
say("x (no inputs at all)", int(no_in.sum())); say("y (no outputs at all)", int(no_out.sum()))
say("z (neither)", int((no_in & no_out).sum()))
say("no input partner at >=10", int((feats["in_degree"] == 0).sum()))
say("no output partner at >=10", int((feats["out_degree"] == 0).sum()))

# ---- Same checks inside the central brain only (your modelling population)
fc = feats.merge(neurons[["neuron_id", "is_optic"]], on="neuron_id"); fc = fc[~fc["is_optic"]]
say("central: no inputs / no outputs / neither", f"{int((fc.total_in_syn==0).sum())} / {int((fc.total_out_syn==0).sum())} / {int(((fc.total_in_syn==0)&(fc.total_out_syn==0)).sum())}")
say("central: no input partner / no output partner at >=10", f"{int((fc.in_degree==0).sum())} / {int((fc.out_degree==0).sum())}")
say("central: total_in_syn median / max / skew", f"{fc.total_in_syn.median():.0f} / {fc.total_in_syn.max():.0f} / {skew(fc.total_in_syn.to_numpy(float)):.2f}")

# ---- Split: neurons without cell_type, and size of biggest family (majority baseline)
say("neurons in kept families with no cell_type", int(c["cell_type"].isna().sum()), "decide: drop or group alone")
keep = fam[(fam.n_neurons >= 5) & (fam.n_cell_types >= 2)]  # example cutoff, change N to your choice
if len(keep):
    W = 100 * keep.n_neurons.max() / keep.n_neurons.sum()
    say("W (% of neurons in the biggest family, example N=5)", round(W, 1), "majority-class accuracy")
open("first_look_numbers.md", "w").write("\n".join(out))
print("\nSaved first_look_numbers.md, table_families.csv, fig1_family_sizes.png, fig2_raw_counts.png")
