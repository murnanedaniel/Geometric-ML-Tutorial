# Geometric Deep Learning Tutorial

Hands-on tutorial materials for the **AI4Physics Learning Workshop**, Uppsala University, 16–17 April 2026.

> *Geometric Deep Learning: Hands-on* — Daniel Murnane (NBI)

## What's Inside

Two self-contained Jupyter notebooks demonstrating Graph Neural Networks (GNNs) applied to two very different physics problems:

| Notebook | Topic | Audience hook |
|---|---|---|
| [`02_Metallic_Glass_GNN.ipynb`](notebooks/02_Metallic_Glass_GNN.ipynb) | GNN for glass-forming ability of metallic alloys, with explicit message passing + CHGNet for structural relaxation | Condensed matter, materials science, metallurgy |
| [`03_Feynman_Diagrams_GNN.ipynb`](notebooks/03_Feynman_Diagrams_GNN.ipynb) | GNN for tree-level QED scattering amplitudes — Feynman diagrams *are* graphs | HEP, QFT, string theory, amplitudes |

The two notebooks are independent — pick whichever matches your interest. Notebook 02 is the recommended starting point because it opens up the "black box" of message passing from scratch; notebook 03 then builds on that foundation and compares GNNs to MLP and Transformer baselines.

---

## Setup

Both notebooks have been tested end-to-end on **Python 3.11** with the exact versions pinned in [`requirements.txt`](requirements.txt). The two notebooks have different dependency stacks, so pick the install route that matches your needs.

### Option A — Google Colab (recommended for most users)

The first code cell of each notebook is a commented `!pip install` line — uncomment it and run. This should Just Work on Colab; no local setup required.

### Option B — Local install (both notebooks)

```bash
git clone https://github.com/murnanedaniel/Geometric-ML-Tutorial.git
cd Geometric-ML-Tutorial

# Create a fresh virtualenv (any of these work)
python3.11 -m venv .venv && source .venv/bin/activate
# — or —
uv venv .venv --python 3.11 && source .venv/bin/activate    # faster
# — or —
conda create -n geom-ml python=3.11 -y && conda activate geom-ml

pip install -r requirements.txt

# Register the venv as a Jupyter kernel so the notebooks can find it
python -m ipykernel install --user --name=geom-ml --display-name="Python (geom-ml)"

jupyter lab
```

In the notebook, select the **"Python (geom-ml)"** kernel from the kernel menu before running. Both notebooks run on CPU; a GPU is optional (and will make notebook 03 a bit faster).

### Option C — Minimal install (one notebook only)

If you only want one of the two notebooks, you can skip the other's dependencies:

```bash
# For notebook 02 (metallic glass) — needs the materials informatics stack:
pip install 'numpy<2' pandas matplotlib scikit-learn tqdm \
            torch torch_geometric \
            'matminer>=0.9.3' 'pymatgen==2024.6.10' 'chgnet>=0.3' 'ase>=3.22' \
            jupyter ipykernel

# For notebook 03 (Feynman diagrams) — needs only PyTorch + PyG + networkx:
pip install numpy pandas matplotlib scikit-learn \
            torch torch_geometric networkx \
            jupyter ipykernel
```

### ⚠️ Important version pins (notebook 02)

Two version pins are **required** for notebook 02:

1. **`numpy<2`** — matminer imports a cython-compiled pymatgen submodule that was built against numpy 1.x. With numpy 2.x it will fail with `numpy.core.multiarray failed to import`.
2. **`pymatgen==2024.6.10`** (or similar) + **`matminer>=0.9.3`** — newer pymatgen versions remove internals matminer depends on (`_pt_data`); and matminer <0.9.3 iterates to `Z=119` which pymatgen doesn't know about.

`requirements.txt` already has these pins correct.

---

## Notebook 02: Metallic Glass Property Prediction *(the "first" notebook)*

**Goal.** Predict which alloy compositions form metallic glasses, using a GNN over compositions + a pretrained universal interatomic potential on 3D structures.

**What makes this notebook special.** We **open up the GNN black box** and implement message passing from scratch — you see the three primitive operations (gather, transform, scatter-add) on a toy 3-node graph, then package them into a `GCNFromScratch` layer that is verified to match `torch_geometric.nn.GCNConv` *exactly* (max absolute difference = 0 on a random toy input). This gives you a mental model you can use for any subsequent GNN.

**Physics content.**
- The glass-forming ability (GFA) problem
- Alloy composition as a small graph (elements = nodes, fully connected)
- 3D crystal phases and their energies (CHGNet universal potential)

**ML content.**
- The three primitive operations of message passing: gather, transform, scatter-add
- Worked example on a tiny 3-node graph, with every intermediate tensor printed
- From-scratch GCN layer, equivalent to PyG's `GCNConv`
- Binary classification with a simple GCN architecture
- Comparison to a Random Forest baseline on hand-crafted Magpie features
- Using pretrained CHGNet for relaxation and energy prediction

**Dataset.** `matbench_glass` from [matminer](https://hackingmaterials.lbl.gov/matminer/) — 5,680 compositions labelled as glass former (4035) or not (1645). Downloads automatically via `matminer.datasets.load_dataset()` the first time you run the notebook (~40 KB).

**Expected results** (verified end-to-end run):
- Toy message-passing walk-through prints every intermediate tensor
- `GCNFromScratch` matches PyG's `GCNConv` with max abs diff = 0
- GlassGNN (18 raw node features, 3 GCN layers, class-weighted BCE): **~78.5% test acc, 0.81 AUC** (non-glass recall 0.53 — not collapsing to majority class)
- Random Forest on 130+ hand-crafted Magpie features: **~90% accuracy, 0.95 AUC** — RF wins on this composition-only task, which is an honest and pedagogically useful outcome (Magpie features *are* the aggregate statistics the GNN is trying to learn; the GNN's payoff would come when structure information is available)
- CHGNet evaluates crystalline metallic phase energies and relaxes a distorted CuZr B2 to convergence in ~14 BFGS steps

**Runtime.** ~10 minutes on CPU.

---

## Notebook 03: Feynman Diagrams as Graphs *(the "second" notebook)*

**Goal.** Show that Feynman diagrams are literally graphs, and that passing messages between their vertices lets a neural network predict $|\mathcal{M}|^2$. Compare against MLP and Transformer baselines to build an honest picture of when graph structure helps.

**Physics content.**
- Tree-level QED 2→2 scattering: $e^+e^-\to\mu^+\mu^-$ (s-channel only) and Bhabha $e^+e^-\to e^+e^-$ (s+t)
- Diagrams as graphs: initial / virtual / final vertices as nodes; propagators with Standard Model quantum numbers (mass, spin, weak isospin, hypercharge, colour) as edge features
- Analytic matrix element formulas — no MadGraph needed

**ML content.**
- Dataset generation from scratch (no external download!)
- Architecture hierarchy of inductive biases:
  - **KinematicMLP** (weak: 5 features only)
  - **FlattenedMLP** (fair: padded graph + log1p normalization, 321 features)
  - **EdgeTransformer** (permutation-equivariant via attention + masking)
  - **FeynmanGNN** (adjacency-restricted attention via `GATConv`)
- Out-of-distribution generalization test across topologies

**Dataset.** Fully self-generated — analytic QED formulas give $|\mathcal{M}|^2$ labels on a grid of $(p, \theta)$ for both processes, ~5000 samples each.

**Expected results** (verified end-to-end run, in-distribution on combined mumu + Bhabha):

| Model | Params | Val MSE |
|---|---|---|
| KinematicMLP     | 8.7K    | **0.70**   (fails — cannot identify the process from kinematics alone) |
| FlattenedMLP     | 214K    | **0.004**  (strong fair baseline with log1p normalization) |
| EdgeTransformer  | 78K     | **0.002**  (permutation-equivariant via attention, fewer params than the flat MLP) |
| FeynmanGNN       | ~20K    | **0.0006** (best in-dist; smallest parameter count of the three strong models) |

**Out-of-distribution (train on $\mu^+\mu^-$ only, test on Bhabha):** mean absolute log-errors in the 2.7–3.2 range, i.e. factor-of-~20 off for all four models. The FlattenedMLP often generalizes *most gracefully* on this toy — an honest reminder that architectural inductive biases are not magic.

**The honest lesson.** For a fixed, small-topology 2→2 tree problem, the four architectures span a hierarchy of inductive biases:
- **MLP on flat** — everything interacts; needs canonical ordering + normalization
- **Transformer on edge set** — permutation-equivariant via attention; handles padding via masks (fewer params than the flat MLP)
- **GNN on graph** — permutation-equivariant AND attention restricted to adjacency

All three "strong" architectures work on this toy. The GNN's real payoff is at scale: many processes, loop diagrams, parameter sharing across topologies. Always build fair baselines before claiming a GNN helps.

**Runtime.** ~5 minutes on CPU.

### Based On

Mitchell, *Learning Feynman Diagrams using Graph Neural Networks*, [arXiv:2211.15348](https://arxiv.org/abs/2211.15348) — code at [matbun/Feynman-GNN](https://github.com/matbun/Feynman-GNN). Our pipeline is a pedagogically simplified rewrite of the data generator and model from that repo.

---

## Further Reading

Each notebook ends with an "Extensions & Related Work" section pointing to:
- For materials: CGCNN, MEGNet, ALIGNN, CHGNet, MACE, NequIP; Materials Project, MatBench
- For amplitudes: SYMBA (transformer for squared amplitudes), L-CNNs for lattice gauge theory, quiver-mutation GNNs for string theory, SDRG-GNNs for disordered spin chains

## Troubleshooting

**`NoSuchKernel: ...` when opening a notebook.** The committed notebooks specify a kernel name that your system may not have; use the kernel menu in Jupyter to pick any Python 3 kernel (or follow the `ipykernel install` step in the Local install above to register one).

**`ImportError: cannot import name '_pt_data'` from pymatgen.** You have a pymatgen version newer than matminer supports. Pin `pymatgen==2024.6.10` (already set in `requirements.txt`).

**`ValueError: Unexpected atomic number Z=119`.** You have `matminer<0.9.3`. Upgrade (`pip install 'matminer>=0.9.3'`).

**`numpy.core.multiarray failed to import`.** You have numpy ≥ 2.x. Downgrade (`pip install 'numpy<2'`).

**`matbench_glass` download retries 3 times then fails.** Transient server issue on `ml.materialsproject.org`. Just re-run the cell. If it persists, manually download `matbench_glass.json.gz` from the Materials Project site and drop it into the matminer datasets directory.

## License

Code: MIT. Text & figures: CC-BY-4.0.
