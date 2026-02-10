# Dreaming in Code for Curriculum Learning in Open-Ended Worlds

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Dreaming in Code (DiCode)** is a framework for Unsupervised Environment Design (UED) that leverages Foundation Models to synthesize executable environment code, creating an adaptive curriculum for agent learning in complex, open-ended worlds.

---

## 🚀 Quick Start

Get running in seconds using [uv](https://github.com/astral-sh/uv).

```bash
# 1. Clone & Install
git clone https://github.com/konstantinosmitsides/DiCode.git
cd DiCode
uv sync --all-extras

# For NVIDIA GPU support (ensure [cuda12] matches your CUDA version):
uv pip install "jax[cuda12]"

# 2. Configure Secrets
cp .env.example .env
# Edit .env and add your secret keys

# 3. Train Agent
uv run experiments/training/run_dicode.py
```

---

## 📦 Installation

### Prerequisites
- **Python**: 3.10+
- **JAX**: Install the version compatible with your hardware (CPU/GPU/TPU).
  ```bash
  # Example for CUDA 12
  uv pip install "jax[cuda12]"
  ```

### Alternative Methods
<details>
<summary><b>Pip fallback</b></summary>

```bash
pip install -e .[dev,evaluation]
```
</details>

<details>
<summary><b>Apptainer</b></summary>

```bash
# Build
apptainer build dicode.sif apptainer/container.def

# Run training
apptainer run --nv dicode.sif

# Interactive shell (Development)
apptainer shell --nv --bind .:/workspace dicode.sif
```
</details>

<details>
<summary><b>Docker</b></summary>

```bash
# Build
docker build -t dicode .

# Run training
docker run --gpus all --env-file .env dicode

# Interactive shell (Development)
docker run --gpus all -it --env-file .env -v $(pwd):/workspace dicode /bin/bash
```
</details>

---

## ⚙️ Configuration & Secrets

DiCode uses environment variables for sensitive configuration like API keys.

1.  **`.env` file**: Copy `.env.example` to `.env` and fill in your keys.
2.  **Automatic Loading**: The project automatically loads the `.env` file upon importing the `dicode` package. You can run scripts directly without manual exports:
    ```bash
    uv run experiments/training/run_dicode.py
    ```

### Hydra Overrides
You can override any parameter via the command line:
```bash
uv run experiments/training/run_dicode.py seed=42 use_wandb=false
```

---
## 📂 Repo Structure

- **`src/dicode`**: Core implementation (Evolution, PPO, LLM Manager).
- **`src/minicraftax`**: Environment and task definitions.
- **`experiments/`**: Training and experimentation entry points.
- **`conf/`**: Hydra configuration files.

---

## 📜 Citation

```bibtex
@misc{mitsides2026dreamingcodecurriculumlearning,
      title={Dreaming in Code for Curriculum Learning in Open-Ended Worlds}, 
      author={Konstantinos Mitsides and Maxence Faldor and Antoine Cully},
      year={2026},
      eprint={2602.08194},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.08194}, 
}
```