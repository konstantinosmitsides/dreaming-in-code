import os
from pathlib import Path

import numpy as np
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from scipy import spatial


def read_file(file_path):
	with open(file_path) as file:
		return file.read()


def smart_absolute_path(path: str) -> str:
	"""Enhanced version of to_absolute_path that handles both:
	1. Generated task paths (relative to current working directory)
	2. Archive paths (relative to the REPOSITORY ROOT, not original cwd)

	First tries current working directory, then falls back to repository root.
	The repo root is found by walking up from original cwd looking for pyproject.toml.
	"""
	p = Path(path)

	# If already absolute, return as is
	if p.is_absolute():
		return str(p)

	# Try current working directory first (for generated tasks in Hydra output dir)
	current_path = Path(os.getcwd()) / p
	if current_path.exists():
		return str(current_path)

	# Fall back to repository root
	# Find repo root by walking up from original cwd looking for pyproject.toml
	if HydraConfig.initialized():
		start_dir = Path(get_original_cwd())
	else:
		start_dir = Path(os.getcwd())
	
	repo_root = _find_repo_root(start_dir)
	original_path = repo_root / p
	return str(original_path)


def _find_repo_root(start_path: Path) -> Path:
	"""Walk up from start_path looking for pyproject.toml to find repo root."""
	current = start_path.resolve()
	while current != current.parent:
		if (current / "pyproject.toml").exists():
			return current
		current = current.parent
	# Fallback to start_path if not found
	return start_path


def get_codet5_embeddings(texts):
	import torch
	from transformers import AutoModel, AutoTokenizer

	checkpoint = "Salesforce/codet5p-110m-embedding"
	device = "cuda" if torch.cuda.is_available() else "cpu"
	tokenizer = AutoTokenizer.from_pretrained(checkpoint)
	model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)
	embeddings = []
	for text in texts:
		inputs = tokenizer.encode(text, return_tensors="pt").to(device)
		embedding = model(inputs)[0].detach().cpu().numpy()
		embeddings.append(embedding)
	return embeddings


def distances_from_embeddings(
	query_embedding: np.ndarray,
	embeddings: list[np.ndarray],
	distance_metric: str = "cosine",
) -> list[float]:
	"""Calculate distances between a query embedding and a list of embeddings."""
	distance_metrics = {
		"cosine": spatial.distance.cosine,
		"L1": spatial.distance.cityblock,
		"L2": spatial.distance.euclidean,
		"Linf": spatial.distance.chebyshev,
	}

	distances = [
		distance_metrics[distance_metric](query_embedding, embedding) for embedding in embeddings
	]
	return distances


# Helper function to find the last saved env file in a directory
def get_last_env_file_index(task_dir):
	env_files = [f for f in os.listdir(task_dir) if f.startswith("env_") and f.endswith(".py")]
	if not env_files:
		return 0
	# Extract numbers from filenames like env_0.py, env_1.py, etc.
	indices = []
	for f in env_files:
		try:
			# Extract number between "env_" and ".py"
			num_str = f[4:-3]  # Remove "env_" prefix and ".py" suffix
			indices.append(int(num_str))
		except ValueError:
			continue
	return max(indices) if indices else 0
