# Geometric Deep Learning Tutorial

Hands-on tutorial materials for the **AI4Physics Learning Workshop**, Uppsala University, 16–17 April 2026.

> *Geometric Deep Learning: Hands-on* — Daniel Murnane (NBI)

## What's Inside

Two self-contained Jupyter notebooks demonstrating Graph Neural Networks (GNNs) applied to two very different physics problems:

| Notebook | Topic | Audience hook |
|---|---|---|
| [`02_Crystal_Structure_GNN.ipynb`](notebooks/02_Crystal_Structure_GNN.ipynb) | GNN for crystal property prediction, with explicit message-passing machinery built from scratch | Condensed matter, materials science, chemistry |
| [`03_Feynman_Diagrams_GNN.ipynb`](notebooks/03_Feynman_Diagrams_GNN.ipynb) | GNN for tree-level QED scattering amplitudes — Feynman diagrams *are* graphs | HEP, QFT, string theory, amplitudes |

The two notebooks are independent — pick whichever matches your interest. Notebook 02 is the recommended starting point because it opens up the "black box" of message passing from scratch; notebook 03 then builds on that foundation and compares GNNs to MLP and Transformer baselines.

---

## 🚀 Quick Start (Google Colab — recommended)

The fastest way to run the notebooks is on Google Colab — no local install required.

1. **Open the notebook**: go to [colab.research.google.com](https://colab.research.google.com/), then `File → Open Notebook → GitHub`, paste this repo URL, and select the notebook you want.
   - Or directly: [![Open Notebook 02 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/murnanedaniel/Geometric-ML-Tutorial/blob/main/notebooks/02_Crystal_Structure_GNN.ipynb) **Crystal structure GNN**
   - Or directly: [![Open Notebook 03 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/murnanedaniel/Geometric-ML-Tutorial/blob/main/notebooks/03_Feynman_Diagrams_GNN.ipynb) **Feynman diagrams GNN**

2. **Uncomment the `!pip install ...` line** in the first code cell and run it. First-run install takes ~1–2 minutes.

3. **Run cells top-to-bottom** — everything is self-contained, with datasets downloaded automatically.

Each notebook runs end-to-end in **5–10 minutes on Colab's CPU**. A GPU runtime is not required but will speed up notebook 03 slightly.

---

## Local Install (alternative)

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

# Register the venv as a Jupyter kernel
python -m ipykernel install --user --name=geom-ml --display-name="Python (geom-ml)"

jupyter lab
```

When opening a notebook, select the **"Python (geom-ml)"** kernel from the kernel menu.

### Minimal install (one notebook only)

```bash
# For notebook 02 (crystal structure) — needs the materials informatics stack:
pip install 'numpy<2' pandas matplotlib scikit-learn \
            torch torch_geometric \
            'matminer>=0.9.3' 'pymatgen==2024.6.10' \
            jupyter ipykernel

# For notebook 03 (Feynman diagrams) — needs only PyTorch + PyG + networkx:
pip install numpy pandas matplotlib scikit-learn \
            torch torch_geometric networkx \
            jupyter ipykernel
```

### ⚠️ Important version pins (notebook 02)

Two version pins are **required** for notebook 02 to work:

1. **`numpy<2`** — matminer imports a cython-compiled pymatgen submodule built against numpy 1.x. With numpy 2.x it fails with `numpy.core.multiarray failed to import`.
2. **`pymatgen==2024.6.10`** + **`matminer>=0.9.3`** — newer pymatgen versions remove internals matminer depends on; and older matminer versions iterate to Z=119 which pymatgen doesn't know about.

`requirements.txt` and the Colab install line already have these pins correct.

---

## Notebook 02: Crystal Structure Property Prediction

**Goal.** Build a Graph Neural Network **from scratch** and apply it to predict formation energies of crystals from their 3D atomic structures.

**Why it's special.** Most GNN tutorials treat `GCNConv` as a black box. This one opens it up — you see the three primitive operations (gather, transform, scatter-add) on a toy 3-node graph, then package them into a `GCNFromScratch` layer verified to match PyTorch Geometric's `GCNConv` **exactly** (max abs diff = 0). Then we scale up to edge-aware message passing (CGCNN-style) for real crystal graphs.

**Physics content.**
- Crystals as graphs: atoms = nodes, bonds within cutoff = edges, distances = edge features
- Graph topology varies meaningfully across crystal types (BCC vs FCC vs perovskite vs ...)
- Formation energy prediction (the canonical crystal GNN benchmark)

**ML content.**
- **Section 2**: message passing from scratch (toy 3-node worked example + `GCNFromScratch` module)
- **Section 3**: loading real crystal structures from the `flla` dataset (3,938 crystals from Materials Project)
- **Section 4**: edge-aware message passing (`EdgeConvFromScratch`, simplified CGCNN)
- **Section 5**: honest baseline comparison vs Random Forest and MLP on composition features

**Dataset.** `flla` from matminer — 3,938 Materials Project structures with DFT-computed formation energies. Downloads automatically (~2.6 MB).

**Expected results** (verified end-to-end run):

| Model | Input | MAE (eV/atom) |
|---|---|---|
| MLP on Magpie features | Composition only | ~1.65 (fails) |
| Random Forest on Magpie | Composition only | ~0.166 |
| **Crystal GNN (ours)** | **3D crystal structure** | **~0.160** |

The GNN beats the strong RF baseline by ~4% because the 3D bonding topology and interatomic distances carry signal that composition-only descriptors miss.

**Runtime.** ~5 minutes on CPU.

---

## Notebook 03: Feynman Diagrams as Graphs

**Goal.** Show that Feynman diagrams are literally graphs, and that message passing between their vertices lets a neural network predict $|\mathcal{M}|^2$. Compare against MLP and Transformer baselines to build an honest picture of when graph structure helps.

**Physics content.**
- Tree-level QED 2→2 scattering: $e^+e^-\to\mu^+\mu^-$ (s-channel) and Bhabha $e^+e^-\to e^+e^-$ (s+t channels)
- Diagrams as graphs: initial / virtual / final vertices as nodes; propagators with Standard Model quantum numbers (mass, spin, weak isospin, hypercharge, colour) as edge features
- Analytic matrix element formulas — no MadGraph needed

**ML content.**
- Dataset generation from scratch (no external download!)
- Architecture hierarchy of inductive biases:
  - **KinematicMLP** (weak: 5 features only)
  - **FlattenedMLP** (fair: padded graph + log1p normalization, 321 features)
  - **EdgeTransformer** (permutation-equivariant via attention + masking)
  - **FeynmanGNN** (adjacency-restricted attention via `GATConv`)

**Dataset.** Fully self-generated — analytic QED formulas give $|\mathcal{M}|^2$ labels on a grid of $(p, \theta)$ for both processes, ~5000 samples each.

**Expected results** (in-distribution on combined mumu + Bhabha):

| Model | Params | Val MSE |
|---|---|---|
| KinematicMLP     | 8.7K    | **0.70**   (fails — cannot identify process from kinematics alone) |
| FlattenedMLP     | 214K    | **0.004**  (strong fair baseline with log1p normalization) |
| EdgeTransformer  | 78K     | **0.002**  (permutation-equivariant via attention) |
| FeynmanGNN       | ~20K    | **0.0006** (best, smallest parameter count) |

**The architectural punchline.** Each step down the hierarchy adds a more aligned inductive bias and *reduces* the parameter count needed to solve the task. That's the geometric deep learning story: the right structure makes the problem easier.

**Runtime.** ~5 minutes on CPU.

### Based On

Mitchell, *Learning Feynman Diagrams using Graph Neural Networks*, [arXiv:2211.15348](https://arxiv.org/abs/2211.15348) — code at [matbun/Feynman-GNN](https://github.com/matbun/Feynman-GNN). Our pipeline is a pedagogically simplified rewrite of the data generator and model from that repo.

---

## Further Reading

Each notebook ends with an "Extensions & Research Frontiers" section pointing to:
- **For crystals**: CGCNN, MEGNet, ALIGNN, CHGNet, MACE, NequIP; Materials Project, MatBench, Matbench Discovery
- **For amplitudes**: SYMBA (symbolic squared amplitudes), SAILIR (self-supervised IBP reduction), spinor-helicity simplification, open research gaps (QGRAF+GNN pipelines, graph generative models for Feynman diagrams)
- **General**: [HEPML Living Review](https://github.com/iml-wg/HEPML-LivingReview), [Geometric Deep Learning Book](https://geometricdeeplearning.com/)

## Troubleshooting

**`NoSuchKernel: ...`** when opening a notebook — the committed notebooks specify a kernel name that your system may not have. Use the kernel menu in Jupyter to pick any Python 3 kernel, or follow the `ipykernel install` step in the Local Install section to register one.

**`ImportError: cannot import name '_pt_data'` from pymatgen** — you have a pymatgen version newer than matminer supports. Pin `pymatgen==2024.6.10` (already set in `requirements.txt`).

**`ValueError: Unexpected atomic number Z=119`** — you have `matminer<0.9.3`. Upgrade: `pip install 'matminer>=0.9.3'`.

**`numpy.core.multiarray failed to import`** — you have numpy ≥ 2.x. Downgrade: `pip install 'numpy<2'`.

**`flla` download fails** — transient issue with `ml.materialsproject.org`. Just re-run the cell. If it persists, manually download `flla.json.gz` from the Materials Project site and drop it into your matminer datasets directory.

## License

Code: MIT. Text & figures: CC-BY-4.0.
