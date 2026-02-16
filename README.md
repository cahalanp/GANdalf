# GANdalf6

> [TODO: One-sentence project summary]

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.x-informational.svg)](#requirements)

GANdalf6 is a molecular-structure generation and inversion workflow built around bispectrum descriptors, PyTorch GAN training, and geometry reconstruction for MD17-style datasets.

## Table of Contents

- [Overview](#overview)
- [Repository Layout](#repository-layout)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data Preparation Pipeline](#data-preparation-pipeline)
- [Training](#training)
- [Inversion / Structure Reconstruction](#inversion--structure-reconstruction)
- [Outputs](#outputs)
- [Configuration](#configuration)
- [Reproducibility Notes](#reproducibility-notes)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## Overview

Core workflow:

1. Convert `.npz` molecular data to `.xyz` (`compile/convert_npz_to_xyz.py`).
2. Split `.xyz` into train/test/val chunks (`compile/compile_xyz.py`).
3. Compute bispectrum descriptors using LAMMPS (`compile/compute_b.py`).
4. Merge chunked descriptor files (`compile/merge_b.py`).
5. Train GAN models (`gan/train.py` or `gan/train_e.py`).
6. Invert descriptors back to 3D coordinates (`gan/invert.py` or `gan/invert_e.py`).

Two GAN variants are available:

- `gan/train.py`: descriptor-based GAN.
- `gan/train_e.py`: descriptor + energy-conditioned GAN with an auxiliary energy model.

## Repository Layout

```text
.
|-- compile/          # Data conversion and descriptor compilation scripts
|-- cobe/             # Dataset/descriptors utilities (LAMMPS-backed bispectrum)
|-- modules/          # Model definitions and global configuration
|-- gan/              # Training, inversion, plotting utilities, checkpoints
|-- data/             # Input/output datasets (xyz, bispec, etc.)
|-- thesis/           # Analysis and plotting scripts
|-- LICENSE
`-- README.md
```

## Requirements

System-level tools:

- Python `3.x` ([TODO: pin exact version tested, e.g. `3.10.x`])
- LAMMPS with Python bindings (`lammps` module)
- [Optional but recommended] CUDA-compatible GPU for training

Python packages used in this codebase include:

- `torch`
- `numpy`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `tqdm`
- `ase`
- `pyscf`
- `mendeleev`
- `dask` and `distributed`
- `rmsd`

[TODO: Add pinned dependency versions in `requirements.txt` or `environment.yml`]

## Installation

```bash
git clone <YOUR_REPO_URL>
cd gandalf6

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch numpy scipy scikit-learn matplotlib seaborn tqdm ase pyscf mendeleev dask distributed rmsd lammps
```

[TODO: Replace with your exact install command once dependencies are pinned]

## Data Preparation Pipeline

Expected dataset directories are based on `modules/config.py` molecule names (e.g., `9_ethanol`, `24_azobenzene`).

1. Place source `.npz` files in `data/xyz/<molecule_name>/`.
2. Convert `.npz` to `.xyz`:

```bash
python compile/convert_npz_to_xyz.py -n 9
```

3. Split into train/test/val chunks:

```bash
python compile/compile_xyz.py -n 9 -d 0 -p 50000 -s 1
```

4. Compute bispectrum on chunked files:

```bash
python compile/compute_b.py -n 9 -s train -x 0 -rc 1.8 -j 8 -u 0
```

5. Merge computed arrays into a final set:

```bash
python compile/merge_b.py -n 9 -s train -p -1 -u 0 -N 10000 -M 100
```

Notes:

- Several scripts can prompt for interactive choices (dataset directory, set size, file index).
- Repeat step 4 for all required chunk indices (`-x`).

## Training

Train base GAN:

```bash
python gan/train.py -mols 9 -resume "" -N_epoch 5000 -gpu 0
```

Train energy-aware GAN:

```bash
python gan/train_e.py -mols 9 -resume "" -N_epoch 5000 -N_epoch_E 1000 -gpu 0
```

Important behavior:

- Both scripts can ask you to select dataset subdirectories (set type and set size).
- Model checkpoints and run metadata are saved under `gan/savednet/<mol_key>/<run_id>/`.

## Inversion / Structure Reconstruction

After training, reconstruct structures from generated descriptors:

```bash
python gan/invert.py -mols 9 -s_G <RUN_ID> -N_inv 100 -device cpu
```

Energy-aware inversion:

```bash
python gan/invert_e.py -mols 9 -s_G <RUN_ID> -N_inv 100 -device cpu
```

[TODO: Document your recommended inversion hyperparameters for production runs]

## Outputs

Typical outputs include:

- Checkpoints and run settings: `gan/savednet/...`
- Training plots and diagnostics: `gan/savednet/.../plots/`
- Inversion starting structures: `gan/invert/start/<molecule>/`
- Inversion trajectories/final geometries: `gan/invert/...`

## Configuration

Global config lives in `modules/config.py`:

- Project base path
- Molecule naming conventions
- Element/species mapping
- Descriptor constants (`RCUTFAC`, `RFAC0`, `TWOJMAX_DICT`)

[TODO: Add a short section on which config values are safe to modify]

## Reproducibility Notes

- Set random seeds in PyTorch/NumPy before training ([TODO: add your canonical seed setup]).
- Record CLI args and git commit hash for every run.
- Keep train/test/val splits fixed when comparing experiments.

## Troubleshooting

- `ModuleNotFoundError: lammps`: install LAMMPS Python bindings and verify your environment path.
- `CUDA out of memory`: reduce `-N_batch`, model widths (`-N_units_*`), or use CPU.
- Slow descriptor generation: lower chunk size in `compile/compile_xyz.py` and parallelize runs externally.
- Inversion instability: adjust learning-rate/step parameters in inversion scripts and verify scaling metadata.

## Roadmap

- [TODO] Add pinned dependency file (`requirements.txt` or `environment.yml`)
- [TODO] Add automated tests for data and model pipelines
- [TODO] Add non-interactive mode for all scripts
- [TODO] Add CI for linting/basic smoke checks

## Contributing

1. Create a feature branch.
2. Keep changes scoped and reproducible.
3. Include validation steps (commands + outputs) in your PR description.

[TODO: Add your preferred coding standards/linting workflow]

## Citation

[TODO: Add publication or thesis citation for this project]

## License

This project is licensed under the GNU General Public License v3.0. See `LICENSE` for details.
