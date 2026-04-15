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

## Quick Start

### Option A — Google Colab (recommended for most users)

Open the notebook you want directly on Colab and uncomment the first `!pip install` cell. No local setup required.

### Option B — Local install

```bash
git clone https://github.com/murnanedaniel/Geometric-ML-Tutorial.git
cd Geometric-ML-Tutorial
python -m venv .venv && source .venv/bin/activate   # or use conda/mamba
pip install -r requirements.txt
jupyter lab
```

Both notebooks run on CPU in a few minutes. A GPU will make notebook 03 faster but isn't required.

---

## Notebook 02: Metallic Glass Property Prediction *(the "first" notebook)*

**Goal.** Predict which alloy compositions form metallic glasses, using a GNN over compositions + a pretrained universal interatomic potential on 3D structures.

**What makes this notebook special.** We **open up the GNN black box** and implement message passing from scratch — you see the three primitive operations (gather, transform, scatter-add) on a toy 3-node graph, then package them into a `GCNFromScratch` layer that is verified to match `torch_geometric.nn.GCNConv` exactly. This gives you a mental model you can use for any subsequent GNN.

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

**Dataset.** `matbench_glass` from [matminer](https://hackingmaterials.lbl.gov/matminer/) — ~5,700 compositions labelled as glass former or not. Downloads automatically via `matminer.datasets.load_dataset()`.

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
  - **FlattenedMLP** (fair: padded graph + log1p normalization)
  - **EdgeTransformer** (permutation-equivariant via attention + masking)
  - **FeynmanGNN** (adjacency-restricted attention via `GATConv`)
- Out-of-distribution generalization test across topologies

**Dataset.** Fully self-generated — analytic QED formulas give $|\mathcal{M}|^2$ labels on a grid of $(p, \theta)$.

**Expected results** (as observed in testing):

**In-distribution (train & val on combined mumu + Bhabha):**

| Model | Params | Val MSE |
|---|---|---|
| KinematicMLP     | 8.7K    | **0.70**   (fails — cannot identify the process from kinematics alone) |
| FlattenedMLP     | 214K    | **0.004**  (strong fair baseline with log1p normalization) |
| EdgeTransformer  | 78K     | **0.002**  (permutation-equivariant via attention, fewer params than the flat MLP) |
| FeynmanGNN       | ~20K    | **0.0006** (best in-dist; smallest parameter count of the three strong models) |

**Out-of-distribution (train on $\mu^+\mu^-$ only, test on Bhabha):**

All four models fail badly on the unseen topology — mean absolute log-errors in the 2.7–3.2 range, i.e., factor-of-~20 off. The FlattenedMLP often generalizes *most gracefully* on this toy problem — an honest reminder that architectural inductive biases are not magic.

**The honest lesson.** For a fixed, small-topology 2→2 tree problem, the four architectures span a hierarchy of inductive biases:
- **MLP on flat** — everything interacts; needs canonical ordering + normalization
- **Transformer on edge set** — permutation-equivariant via attention; handles padding via masks (fewer params than the flat MLP)
- **GNN on graph** — permutation-equivariant AND attention restricted to adjacency

All three "strong" architectures work on this toy. The GNN's real payoff is at scale: many processes, loop diagrams, parameter sharing across topologies. Always build fair baselines before claiming a GNN helps.

### Based On

Mitchell, *Learning Feynman Diagrams using Graph Neural Networks*, [arXiv:2211.15348](https://arxiv.org/abs/2211.15348) — code at [matbun/Feynman-GNN](https://github.com/matbun/Feynman-GNN). Our pipeline is a pedagogically simplified rewrite of the data generator and model from that repo.

---

## Further Reading

Each notebook ends with an "Extensions & Related Work" section pointing to:
- For materials: CGCNN, MEGNet, ALIGNN, CHGNet, MACE, NequIP; Materials Project, MatBench
- For amplitudes: SYMBA (transformer for squared amplitudes), L-CNNs for lattice gauge theory, quiver-mutation GNNs for string theory, SDRG-GNNs for disordered spin chains

## License

Code: MIT. Text & figures: CC-BY-4.0.
