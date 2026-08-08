<div align="center">

# Dreaming in Code for Curriculum Learning in Open-Ended Worlds

[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2602.08194)
[![Project Website](https://img.shields.io/badge/Project-Website-blue?style=for-the-badge&logo=google-chrome&logoColor=white)](https://konstantinosmitsides.github.io/dreaming-in-code)

<br>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-Accelerated-9cf?style=flat-square)](https://github.com/google/jax)

<br>

*Foundation Models that "dream" and materialize executable environment code to scaffold learning in open-ended worlds.*

</div>

---

## 💡 What is this?

**Dreaming in Code (DiCode)** is an Unsupervised Enviornment Design framework that uses Foundation Models (FMs) to generate **executable Python code** for training environments (or levels). Instead of just randomizing parameters, DiCode writes the logic itself – creating a curriculum of distinct levels that bridge the gap between an agent's current skills and the complexities of open-ended worlds.

<div align="center">
<br/>
<img src="assets/method_figure.png" width="100%" alt="DiCode Method Overview">
<br/>
</div>

The framework operates in a closed feedback loop:
1.  **Dream:** An FM synthesizes new environment code (transition dynamics, initial states, goals) tailored to the agent's current capabilities.
2.  **Evaluate:** The agent is trained on these generated levels; performance data flows back into the archive.
3.  **Refine:** High-learning-signal levels become "parents" for the next generation of code, creating an infinite, self-correcting curriculum.

---

## 📈 Key Results

<div align="center">
<img src="assets/learning_curve.png" width="80%" alt="Learning Curve Comparison">
</div>

> **SOTA Performance on [Craftax](https://github.com/MichaelTMatthews/Craftax):** DiCode dominates throughout training, achieving a **16% improvement** in mean return over the strongest baseline (PPO-GTrXL).

By structuring the curriculum through code generation, DiCode:
* **Solves the "Impossible":** Achieves non-zero success on late-game tasks (e.g., *Defeat Gnome Warrior*, *Defeat Gnome Archer*) where baselines fail completely (**0% success**).
* **Unlocks Exploration:** Scaffolds instrumental milestones (e.g., *Make Iron Armour*), enabling the agent to survive long enough to reach and master deep exploration targets.

### 🎥 Watch Gameplay Comparison

| RL Baseline (PPO-GTrXL) on Craftax | DiCode Agent (Ours) on Craftax |
| :---: | :---: |
| <img src="assets/baseline_gameplay.gif" width="100%" alt="Baseline Gameplay"> | <img src="assets/dicode_gameplay.gif" width="100%" alt="DiCode Gameplay"> |
| *Struggles with initial survival.* | *Reaches late-game content.* |

---

## 🚀 Quick Start

Get running in seconds using [uv](https://github.com/astral-sh/uv).

```bash
# 1. Clone & Install
git clone https://github.com/konstantinosmitsides/dreaming-in-code.git
cd dreaming-in-code
uv sync --all-extras

# 2. Install JAX (Ensure [cuda12] matches your driver)
uv pip install "jax[cuda12]"

# 3. Configure Secrets
cp .env.example .env
# Edit .env to add your API keys

# 4. Train Agent
uv run experiments/training/run_dicode.py
```

## ⚙️ Advanced Setup

<details>
<summary><b>Apptainer & Docker Support</b></summary>

**Apptainer (Singularity)**
```bash
# Build
apptainer build dicode.sif apptainer/container.def

# Run training
apptainer run --nv dicode.sif

# Interactive shell
apptainer shell --nv --bind .:/workspace dicode.sif
```
**Docker**
```bash
# Build
docker build -t dicode .

# Run training
docker run --gpus all --env-file .env dicode

# Interactive shell (Development)
docker run --gpus all -it --env-file .env -v $(pwd):/workspace dicode /bin/bash

```

</details>

<details>
<summary><b>Pip / Standard Install</b></summary>

If you prefer not to use [uv](https://github.com/astral-sh/uv), you can install via pip:

```bash
# 1. Install package and dependencies
pip install -e .[dev,evaluation]

# 2. Install JAX (Ensure [cuda12] matches your driver)
pip install "jax[cuda12]"
```

</details>

<details>
<summary><b>Configuration & Hydra Overrides</b></summary>

DiCode uses [Hydra](https://hydra.cc/) for configuration. You can override any parameter from the command line:

```bash
# Make seed random, disable WandB
uv run experiments/training/run_dicode.py seed=$RANDOM use_wandb=false

```

</details>

---

## 🔬 Reproducing the Paper's Baselines

<details>
<summary><b>Training commands, hyperparameters, evaluation protocol and seeds</b></summary>

All four comparison methods from the paper live here, sharing DiCode's PPO-GTrXL agent. Run every command from the repository root.

| Method | Command |
|---|---|
| **PPO-GTrXL** (non-curriculum reference) | `python experiments/training/ppo_gtrxl.py seed=<SEED>` |
| **PLR** | `python -m baselines.plr --seed <SEED>` |
| **DR** | `python -m baselines.plr --seed <SEED> --replay_prob 0.0` |
| **SFL** | `python -m baselines.sfl --seed <SEED>` |

Every remaining hyperparameter is already the script default, and those defaults are the values the published runs used — you should not need to pass anything else to reproduce the paper's setup. DR is not a separate implementation: it is PLR with the replay mechanism disabled, which is exactly how it was run.

To log to Weights & Biases, add `--wandb_entity <your-entity> --project <your-project>` (or `wandb_entity=... wandb_project=...` for the Hydra-based PPO-GTrXL script). Set `WANDB_MODE=disabled` to turn logging off entirely.

**Shared agent and hyperparameters**

All methods use the same Gated Transformer-XL policy, imported from `src/dicode/transformer/transformerXL.py` — the baselines and DiCode load the identical module, so the architecture is shared by construction rather than by duplication.

The shared PPO configuration (Table 6 in the paper): learning rate `2e-4`, 1024 environments × 128 steps, 4 update epochs, 8 minibatches, γ `0.999`, GAE λ `0.8`, clip `0.2`, entropy coefficient `0.002`, value coefficient `0.5`, max gradient norm `1.0`. GTrXL: embedding 256, QKV 256, 8 heads, 2 layers, hidden 256, ReLU, memory window 128, gradient window 64, gating enabled with bias `2.0`.

**Learning rate schedule.** All methods anneal linearly from `2e-4`. DiCode floors at `min_lr = 2e-6`; the baselines decay to `0`.

**Evaluation**

Checkpoints are scored by a single harness, identical for every method including DiCode:

```bash
python experiments/evaluation/run_paper_evaluation.py \
    --results-dir <checkpoint-tree> --cache-dir <cache> --output results.pkl
```

It expects `<results-dir>/<METHOD>/<SEED>/rl_checkpoints/<UPDATE_STEP>/`, evaluates from a fixed `jax.random.PRNGKey(0)`, and reports mean return and per-achievement success rates on the held-out test set — 1024 procedurally generated worlds × 8192 steps, none of which any method trains on. Results are cached per checkpoint, so the scan is resumable.

The paper's runs used `conditioning_type: one_hot`, which is computed locally. No API key or embedding server is needed to evaluate.

**Seeds**

Eight seeds per method:

| Method | Seeds |
|---|---|
| PPO-GTrXL | 188, 5332, 5401, 12794, 16056, 21846, 28462, 31331 |
| PLR | 512, 2215, 2907, 6194, 19943, 21939, 24657, 26080 |
| SFL | 173, 2522, 3286, 6082, 6543, 9067, 11690, 19974 |
| DR | 3041, 7322, 8390, 8401, 14219, 15545, 27525, 30381 |
| DiCode | 42, 5329, 10268, 13438, 28586, 30361, 31249, 31498 |

> **On reproducing DiCode specifically:** fixing the seed does not make a DiCode run deterministic. The curriculum is generated by a foundation model, and that generation phase remains stochastic, so a rerun will produce a different curriculum and will not match the published numbers checkpoint-for-checkpoint. The baselines have no such component and are seed-deterministic.

**Attribution**

The PLR, DR and SFL implementations are derived from [NCC-UED](https://github.com/nmonette/NCC-UED) (Apache-2.0) and are covered by `baselines/LICENSE`, not the MIT license of the rest of this repository. `baselines/NOTICE` records the upstream credits and the modifications made for this paper.

</details>

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