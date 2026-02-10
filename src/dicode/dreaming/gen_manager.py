"""Task evolution and curriculum generation using LLM-based dreaming.

This module implements the core curriculum learning loop for DiCode, including:
- Task: Loading and managing individual task environments.
- TaskArchive: A graph-based archive for storing and querying tasks.
- TaskSelector: Strategies for selecting parent tasks for evolution.
- TaskGenerator: LLM-based generation of new task descriptions.
- EnvGenerator: LLM-based code generation with compilation validation.
- GenManager: The main orchestrator class for the evolution pipeline.
"""

# --- Standard Library ---
import copy
import importlib.util
import json
import os
import re
import sys
import tempfile
import threading
import traceback
from textwrap import dedent

# --- Third-Party ---
import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams

# --- Local Modules ---
from dicode.dreaming.llm import LLM
from dicode.dreaming.prompts.dicode.constants import context as CONSTANTS
from dicode.dreaming.prompts.dicode.minicraftax_api import context as API_DOCS
from dicode.dreaming.prompts.dicode.mobs import context as MOBS
from dicode.dreaming.prompts.dicode.mobs_code import context as MOBS_CODE
from dicode.dreaming.prompts.dicode.step_fn_nl import context as GAME_MECHANICS
from dicode.dreaming.prompts.dicode.world_gen_nl import context as WORLD_GEN
from dicode.dreaming.utils import distances_from_embeddings, smart_absolute_path
from minicraftax.envs.base import MiniCraftaxTrain

# Instruction for the embedding model to generate task embeddings.
EMBEDDING_INSTRUCTION = (
    "Generate an embedding for this Craftax task description to evaluate its "
    "conceptual similarity to other tasks. The embedding should capture the core "
    "gameplay loop, the primary skills the agent must use (e.g., navigation, "
    "crafting, combat), the overall strategic objective, and how the world is built."
)


class Task:
    """Loads and wraps a task environment from a Python file.

    Attributes:
        path: Absolute path to the task's Python file.
        file: The raw source code of the task file.
        env: The wrapped MiniCraftaxTrain environment.
        task: The raw Env class instance.
        desc: The task's docstring description.
    """

    def __init__(self, path: str):
        """Initializes a Task by loading its environment from a file.

        Args:
            path: Absolute path to the task's Python file.
        """
        self.path = path

        with open(self.path) as file:
            self.file = file.read()

        self.env, self.task = self.load_env()
        doc = self.task.__doc__
        self.desc = dedent(doc).strip() if doc else ""

    def load_env(self) -> tuple:
        """Loads the environment class from the task file.

        Uses a unique module name based on the filename to ensure thread safety
        when loading multiple tasks in parallel (sys.modules is shared).

        Returns:
            A tuple of (MiniCraftaxTrain env, raw Env task instance).
        """
        module_name = os.path.splitext(os.path.basename(self.path))[0]
        spec = importlib.util.spec_from_file_location(module_name, self.path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        task = getattr(module, "Env")(
            static_params=StaticEnvParams(), params=EnvParams()
        )
        env = MiniCraftaxTrain(task=task)
        return env, task


class TaskArchive:
	"""Manages the NetworkX graph of tasks, which serves as the single source of
	truth for the curriculum. Handles loading, saving, and querying tasks.
	"""

	def __init__(self, config):
		"""Initializes the TaskArchive by loading an existing graph or creating a new one.

		Args:
		    config: The Hydra configuration object, used to find seed task paths.

		"""
		self.config = config
		self.graph, self.active_task_count = self.load_graph()
		self._lock = threading.Lock()

	def load_graph(self) -> tuple[nx.DiGraph, int]:
		"""Loads the task graph from a file. If no file exists, creates a new
		graph and populates it with the initial seed tasks from the config.

		Returns:
		    A tuple containing the (nx.DiGraph, active_task_count)

		"""
		graph_path = self.config.graph_path  # The standard file to save our graph

		if os.path.exists(graph_path):
			print(f"Loading existing task graph from {graph_path}...")
			g = nx.read_graphml(graph_path)

			active_count = 0
			for node, data in g.nodes(data=True):
				if "performance_history" in data:
					try:
						data["performance_history"] = json.loads(data["performance_history"])
					except (json.JSONDecodeError, TypeError):
						# If it fails, default to an empty list for safety.
						data["performance_history"] = []
				if (
					"is_active" not in data
					or data["is_active"] == "false"
					or data["is_active"] == False
				):
					data["is_active"] = False
				else:
					data["is_active"] = True
					active_count += 1  # Count loaded active tasks

				if "priority_score" not in data:
					data["priority_score"] = float(data.get("learnability_score", 0.0))
				else:
					data["priority_score"] = float(data["priority_score"])

				if "session_last_trained" not in data:
					data["session_last_trained"] = -1
				else:
					data["session_last_trained"] = int(data["session_last_trained"])

			print(f"    - Found {active_count} active tasks in loaded graph.")
			return g, active_count
		else:
			print("No graph file found. Creating new task graph from seed tasks...")
			g = nx.DiGraph()

			# Get the initial task paths from the hydra config
			for i, task_path in enumerate(self.config.example_paths):
				# Add each seed task as a node with initial attributes
				try:
					task = Task(smart_absolute_path(task_path))
					code = task.file
					desc = task.desc
					g.add_node(
						f"task_{i + 1}",
						status="seed",  # "seed" is a special type of success
						type="seed",
						description=desc,
						code=code,
						performance_history=[],
						session_created=0,  # Initialize empty list for metrics,
						is_active=False,
						priority_score=0.0,
						session_last_trained=-1,
					)
				except Exception as e:
					print(f"Warning: Could not load seed task {task_path}. Error: {e}")

			print(f"Created a new graph with {g.number_of_nodes()} seed tasks.")
			return g, 0

	def save_graph(self):
		"""Saves the graph, converting lists to JSON strings for GraphML compatibility."""
		graph_path = "task_graph.graphml"
		print(f"Saving task graph with {self.graph.number_of_nodes()} nodes to {graph_path}...")

		# Create a copy to avoid modifying the graph object that's currently in use.
		with self._lock:
			graph_to_save = copy.deepcopy(self.graph)

		# Loop through all nodes to find and convert JAX arrays
		for node, data in graph_to_save.nodes(data=True):
			if "performance_history" in data and isinstance(data["performance_history"], list):
				# --- NEW: Convert JAX arrays inside the history list ---
				converted_history = []
				for record in data["performance_history"]:
					converted_record = {}
					for key, value in record.items():
						# If the value is a JAX array, convert it to a float
						if isinstance(value, jax.Array):
							converted_record[key] = float(value)
						else:
							converted_record[key] = value
					converted_history.append(converted_record)

				# Now, serialize the cleaned list to a JSON string
				data["performance_history"] = json.dumps(converted_history)

		nx.write_graphml(graph_to_save, graph_path)

	def record_new_task(
		self, child_task: str, parent_tasks: list, description: str, session_id: int
	):
		"""Adds a new task and its parent edges to the graph."""
		with self._lock:
			if not self.graph.has_node(child_task):
				self.graph.add_node(
					child_task,
					status="desc_generated",  # New, more descriptive initial status
					type="generated",
					description=description,
					code="",  # Store code as an attribute
					performance_history=[],
					session_created=session_id,
					is_active=False,
					priority_score=0.0,
					session_last_trained=-1,
				)

			for parent_task in parent_tasks:
				if self.graph.has_node(parent_task):
					self.graph.add_edge(parent_task, child_task)
				else:
					print(f"Warning: Parent task {parent_task} not found in graph.")

	def update_node_status(self, task_path: str, status: str):
		"""Updates the status of a task node (e.g., 'success', 'boring', 'failed_compile').

		Args:
		    task_path: The path of the node to update.
		    status: The new status string.

		"""
		with self._lock:
			if self.graph.has_node(task_path):
				self.graph.nodes[task_path]["status"] = status
			else:
				print(f"Warning: Tried to update status for non-existent node {task_path}")

	def update_node_performance(self, session_idx: int, performance_data: dict):
		"""Updates the performance_history attribute for multiple tasks in the graph.

		Args:
		    session_idx: The current session number.
		    performance_data: A dict mapping task_path to its metrics, e.g.,
		                      {"path/to/task": {"success_rate": 0.8}}.

		"""
		with self._lock:
			for task_path, metrics in performance_data.items():
				if self.graph.has_node(task_path):
					# Ensure the performance_history list exists before appending
					if "performance_history" not in self.graph.nodes[task_path]:
						self.graph.nodes[task_path]["performance_history"] = []

					# Append the new performance record to the node's history
					self.graph.nodes[task_path]["performance_history"].append(
						{"session": session_idx, **metrics}
					)
				else:
					print(f"Warning: Tried to update performance for non-existent node {task_path}")

	def update_node_code(self, task_path: str, code: str):
		"""Sets the env_path attribute for a given task node."""
		with self._lock:
			if self.graph.has_node(task_path):
				self.graph.nodes[task_path]["code"] = code
			else:
				print(f"Warning: Tried to update code for non-existent node {task_path}")

	def get_tasks_by_status(self, statuses: list[str]) -> list[str]:
		"""Queries the graph and returns a list of task paths matching the given statuses.

		Args:
		    statuses: A list of status strings to filter by (e.g., ["seed", "success"]).

		Returns:
		    A list of task path strings (the node IDs).

		"""
		with self._lock:
			return [n for n, d in self.graph.nodes(data=True) if d.get("status") in statuses]

	def get_node_attributes(self, task_path: str) -> dict:
		"""Returns the full attribute dictionary for a given task node.
		Returns an empty dict if the node doesn't exist.
		"""
		with self._lock:
			if self.graph.has_node(task_path):
				return self.graph.nodes[task_path].copy()
			return {}

	def is_active(self, node: str) -> bool:
		"""Checks if a task is currently in the active training set."""
		with self._lock:
			if not self.graph.has_node(node):
				return False
			return self.graph.nodes[node].get("is_active", False)

	def get_task_descriptions(self, task_paths: list[str]) -> dict[str, str]:
		"""Efficiently retrieves the descriptions for a list of tasks from the
		graph's node attributes, avoiding repeated file I/O.

		Args:
		    task_paths: A list of task paths to get descriptions for.

		Returns:
		    A dictionary mapping each task path to its description string.

		"""
		with self._lock:
			descriptions = {}
			for path in task_paths:
				if self.graph.has_node(path):
					# .get("description", "") provides a default empty string if the attr is missing
					descriptions[path] = self.graph.nodes[path].get("description", "")
			return descriptions

	def get_task_codes(self, task_paths: list[str]) -> dict[str, str]:
		"""Efficiently retrieves the codes for a list of tasks from the
		graph's node attributes, avoiding repeated file I/O.

		Args:
		    task_paths: A list of task paths to get codes for.

		Returns:
		    A dictionary mapping each task path to its code string.

		"""
		with self._lock:
			codes = {}
			for path in task_paths:
				if self.graph.has_node(path):
					# .get("description", "") provides a default empty string if the attr is missing
					codes[path] = self.graph.nodes[path].get("code", "")
			return codes

	def get_max_session_idx(self) -> int:
		"""Scans the entire graph to find the highest session index recorded.

		This is useful for resuming an experiment to ensure the session counter
		continues from where it left off.

		Returns:
		    The highest session index found, or 0 if no sessions are recorded.

		"""
		with self._lock:
			max_idx = 0

			# Iterate through every node and its attributes in the graph
			for _, data in self.graph.nodes(data=True):
				# Check the session in which the task was created
				created_idx = data.get("session_created", 0)
				max_idx = max(max_idx, created_idx)

				# Check all session records in the task's performance history
				history = data.get("performance_history", [])
				for record in history:
					perf_idx = record.get("session", 0)
					max_idx = max(max_idx, perf_idx)

			return max_idx

	def get_evolutionary_path(self, task_id: str) -> list[dict]:
		"""Traces the direct ancestral path of a task from a root to the task itself.

		Args:
		    task_id: The ID of the node to trace back from.

		Returns:
		    A list of dictionaries, where each dictionary contains the details
		    of a task in the evolutionary path, ordered from oldest to newest.

		"""
		with self._lock:
			if not self.graph.has_node(task_id):
				return []

			path = []
			current_node_id = task_id

			# Traverse backwards until a node with no parents (a seed) is found
			while True:
				node_data = self.graph.nodes[current_node_id]

				# Extract the most recent performance record
				performance_history = node_data.get("performance_history", [])
				latest_sr = "N/A"
				if performance_history:
					# The history is appended, so the last entry is the most recent
					last_record = performance_history[-1]
					if "sr" in last_record and last_record["sr"] >= 0:
						latest_sr = f"{last_record['sr']:.2%}"  # Format as percentage

				path.append(
					{
						"id": current_node_id,
						"description": node_data.get("description", "No description available."),
						"status": node_data.get("status", "unknown"),
						"success_rate": latest_sr,
					}
				)

				# Get the predecessors of the current node
				predecessors = list(self.graph.predecessors(current_node_id))
				if not predecessors:
					break  # Stop if we've reached a root/seed node

				# In a simple evolution, we assume one parent. If multiple, we follow the first one.
				current_node_id = predecessors[0]

			# The path was built backwards, so we reverse it to get chronological order
			return path[::-1]

	def mark_as_re_evaluated(self, task_id: str):
		"""Sets a flag on a node to indicate it has been re-introduced for training."""
		with self._lock:
			if self.graph.has_node(task_id):
				# The 're_evaluated' attribute defaults to False if not present.
				self.graph.nodes[task_id]["re_evaluated"] = True
			else:
				print(f"Warning: Tried to mark non-existent node {task_id} as re-evaluated.")

	def update_node_learnability(self, task_id: str, score: float):
		"""Updates the learnability score for a task."""
		with self._lock:
			if self.graph.has_node(task_id):
				# Ensure score is a standard float, handle NaN/inf if necessary
				safe_score = float(score) if np.isfinite(score) else 0.0
				self.graph.nodes[task_id]["learnability_score"] = safe_score
			else:
				print(f"Warning: Tried to update learnability for non-existent node {task_id}")

	def remove_node(self, task_id: str):
		"""Removes a task node and its associated edges from the graph."""
		with self._lock:
			if self.graph.has_node(task_id):
				try:
					self.graph.remove_node(task_id)
					print(f"    - Node {task_id} removed from archive.")
				except Exception as e:
					print(f"    - Warning: Failed to remove node {task_id}. Error: {e}")
			else:
				print(f"    - Warning: Tried to remove non-existent node {task_id}.")

	def set_task_active_status(self, task_id: str, is_active: bool):
		"""Atomically sets the 'is_active' status of a task and updates the global counter."""
		with self._lock:
			if not self.graph.has_node(task_id):
				print(f"Warning: Tried to set active status for non-existent node {task_id}")
				return

			current_status = self.graph.nodes[task_id].get("is_active", False)

			if is_active and not current_status:
				# Activate: Set to True and increment counter
				self.graph.nodes[task_id]["is_active"] = True
				self.active_task_count += 1
			elif not is_active and current_status:
				# Deactivate: Set to False and decrement counter
				self.graph.nodes[task_id]["is_active"] = False
				self.active_task_count -= 1
				# Ensure count never goes below zero
				self.active_task_count = max(0, self.active_task_count)
			# else: no change needed (e.g., setting True when already True)

	def update_node_priority_score(self, task_id: str, score: float):
		"""Updates the priority_score for a task."""
		with self._lock:
			if self.graph.has_node(task_id):
				self.graph.nodes[task_id]["priority_score"] = float(score)
			else:
				print(f"Warning: Tried to update priority_score for non-existent node {task_id}")

	def update_node_session_last_trained(self, task_id: str, session_idx: int):
		"""Updates the session_last_trained for a task."""
		with self._lock:
			if self.graph.has_node(task_id):
				self.graph.nodes[task_id]["session_last_trained"] = int(session_idx)
			else:
				print(
					f"Warning: Tried to update session_last_trained for non-existent node {task_id}"
				)

	def update_node_reasoning(self, task_id: str, reasoning: str):
		"""Sets the reasoning attribute for a given task node."""
		with self._lock:
			if self.graph.has_node(task_id):
				self.graph.nodes[task_id]["reasoning"] = reasoning
			else:
				print(f"Warning: Tried to update reasoning for non-existent node {task_id}")


class TaskSelector:
	"""Handles the strategy for selecting which tasks to use as examples for generation.
	Provides methods to select based on task descriptions or full environment code.
	"""

	def __init__(self, archive: TaskArchive, embedding_model: LLM, config):
		"""Initializes the TaskSelector.

		Args:
			archive: The TaskArchive instance for accessing task data.
			embedding_model: LLM instance for generating embeddings.
			config: The Hydra configuration object.
		"""
		self.archive = archive
		self.embedding_model = embedding_model
		self.config = config
		self.example_usage_counts = self._initialize_example_usage_counts()

	def _initialize_example_usage_counts(self) -> dict[str, int]:
		"""Initializes the usage counts based on historical success, measured by
		the number of children (outgoing edges) each task has.
		"""
		print("Initializing example counts from historical success (graph edges)...")
		counts = {}
		for node in self.archive.graph.nodes():
			# The count is the number of successful children
			counts[node] = self.archive.graph.out_degree(node)
		return counts

	def select_similar_desc_tasks(
		self, pivot_task: str, statuses: list[str], num_examples: int
	) -> list[str]:
		"""Selects tasks similar to a pivot based on description embeddings.

		Args:
			pivot_task: The task ID to find similar tasks for.
			statuses: List of valid task statuses to consider.
			num_examples: Number of similar tasks to return.

		Returns:
			A list of task IDs ordered by similarity to the pivot.
		"""
		candidate_tasks = self.archive.get_tasks_by_status(statuses=statuses)
		if not candidate_tasks:
			print("Warning: No successful tasks in archive to select description examples from.")
			return []
		similar_tasks = self._order_similar_tasks(pivot_task, candidate_tasks)
		similar_tasks = similar_tasks[:num_examples]

		self._update_usage_counts(similar_tasks)
		return similar_tasks

	def select_pivot_task(
		self, statuses: list[str], not_use_tasks: list[str], sampling_method: str
	) -> str:
		"""Selects a pivot task for evolution using the specified sampling method.

		Args:
			statuses: List of valid task statuses to consider.
			not_use_tasks: List of task IDs to exclude from selection.
			sampling_method: Either 'frequency' (inverse frequency sampling) or 'random'.

		Returns:
			A task ID string, or an empty list if no candidates available.
		"""
		candidate_tasks = self.archive.get_tasks_by_status(statuses=statuses)
		candidate_tasks = [task for task in candidate_tasks if task not in not_use_tasks]
		if not candidate_tasks:
			print("Warning: No successful tasks in archive to select description examples from.")
			return []
		if sampling_method == "frequency":
			counts = np.array([self.example_usage_counts.get(path, 0) for path in candidate_tasks])
			inv_counts = 1.0 / (counts + 1)
			probabilities = inv_counts / inv_counts.sum()
			return np.random.choice(candidate_tasks, p=probabilities)
		elif sampling_method == "random":
			return np.random.choice(candidate_tasks)
		else:
			raise ValueError(f"Invalid sampling method: {sampling_method}")

	def _update_usage_counts(self, selected_tasks: list[str]):
		"""Updates usage counts for a list of selected tasks."""
		for task in selected_tasks:
			self.example_usage_counts[task] = self.example_usage_counts.get(task, 0) + 1

	def _order_similar_tasks(self, pivot_task: str, other_tasks: list[str]) -> list[str]:
		"""Orders a list of tasks based on their task description similarity to a pivot task description."""
		# Get descriptions directly from the graph attributes
		contents = self.archive.get_task_descriptions(other_tasks)
		pivot_content = contents.get(pivot_task, "")

		# Filter to only include tasks for which we found content
		valid_tasks = list(contents.keys())
		valid_contents = list(contents.values())

		# Get embeddings
		results = self.embedding_model.get_embedding(
			[pivot_content] + valid_contents, EMBEDDING_INSTRUCTION
		)

		pivot_vector = results[0]["embedding"]
		other_vectors = [result["embedding"] for result in results[1:]]

		# Calculate similarity and find top N
		similarities = distances_from_embeddings(
			pivot_vector, other_vectors, distance_metric="cosine"
		)
		sorted_indices = np.array(similarities).argsort()

		return [valid_tasks[i] for i in sorted_indices]


class TaskGenerator:
	"""Handles the creative process of generating new task descriptions using an LLM,
	based on selected examples and agent performance feedback.
	"""

	def __init__(
		self, task_generator_llm: LLM, archive: TaskArchive, selector: TaskSelector, config
	):
		"""Initializes the TaskGenerator.

		Args:
		    task_generator_llm: An instance of the LLM class for generation.
		    archive: An instance of the TaskArchive.
		    config: The Hydra configuration object.

		"""
		self.llm = task_generator_llm
		self.archive = archive
		self.selector = selector
		self.config = config
		if config.mode != "reward":
			self.evolve_mastered_prompt = importlib.import_module(
				self.config.prompts.evolve_mastered
			)
		else:
			self.evolve_mastered_prompt = importlib.import_module(
				self.config.prompts.evolve_mastered_r
			)

		self.ablation_prompt = importlib.import_module(
			self.config.prompts.ablation
		)

		self.task_num_counter = self._initialize_task_counter()

	def _initialize_task_counter(self) -> int:
		"""Finds the highest task number from the archive to avoid overwriting."""
		max_num = -1
		task_paths = [n for n, d in self.archive.graph.nodes(data=True)]
		for path in task_paths:
			match = re.search(r"task_(\d+)", path)
			if match:
				max_num = max(max_num, int(match.group(1)))
		return max_num + 1

	def evolve_ablation(
		self, session_idx: int, mastered_tasks: list[str], global_agent_profile: dict | None = None
	) -> list[dict]:
		"""Generates new task descriptions for ablation experiments.

		This is a simplified version of `evolve_mastered` that uses a fixed prompt
		without agent performance context. Used for ablation studies where we
		want to remove the influence of performance-guided evolution.

		Args:
			session_idx: Current curriculum session index.
			mastered_tasks: List of task IDs the agent has mastered.
			global_agent_profile: Unused in ablation; kept for API compatibility.

		Returns:
			A list of dictionaries containing the generated task data.
		"""
		print(f"Generating {len(mastered_tasks)} new task descriptions (ablation mode)...")

		user_prompts = []
		parent_sets = []
		example_sets = []

		for mastered_task in mastered_tasks:
			task_examples = self.selector.select_similar_desc_tasks(
				mastered_task,
				statuses=self._get_valid_parent_statuses(),
				num_examples=self.config.num_examples,
			)
			parent_sets.append([mastered_task])
			example_sets.append(task_examples)
			# NOTE: Ablation mode uses an empty format() call (no context variables)
			user_prompts.append(self.ablation_prompt.user_prompt.format())

		if not user_prompts:
			print("Could not generate any prompts. Skipping task generation.")
			return []

		system_prompt = self._build_system_prompt(self.ablation_prompt)
		parsed_responses = self._query_and_parse_responses(system_prompt, user_prompts)

		return self._organize_data(
			parsed_responses, parent_sets, example_sets, session_idx, "mastered"
		)

	def evolve_mastered(
		self, session_idx: int, mastered_tasks: list[str], global_agent_profile: dict | None = None
	) -> list[dict]:
		"""Generates new task descriptions by evolving from mastered tasks.

		This is the main evolution method that creates new curriculum tasks based on
		previously mastered tasks and the agent's current performance profile.

		Args:
			session_idx: Current curriculum session index.
			mastered_tasks: List of task IDs the agent has mastered.
			global_agent_profile: Dictionary of agent skill metrics from evaluation.

		Returns:
			A list of dictionaries containing the generated task data.
		"""
		print(f"Generating {len(mastered_tasks)} new task descriptions...")

		user_prompts = []
		parent_sets = []
		example_sets = []

		global_profile_str = self._format_global_agent_profile(global_agent_profile)

		for mastered_task in mastered_tasks:
			task_examples = self.selector.select_similar_desc_tasks(
				mastered_task,
				statuses=self._get_valid_parent_statuses(),
				num_examples=self.config.num_examples,
			)

			parent_sets.append([mastered_task])
			example_sets.append(task_examples)

			# Build the user prompt with performance context
			example_str = self._format_file_mastered_task([mastered_task])
			task_performance_str = self._get_task_performance_str(mastered_task)

			if self.config.mode != "reward":
				user_prompts.append(
					self.evolve_mastered_prompt.user_prompt.format(
						MASTERED_TASK=example_str,
						TASK_PERFORMANCE_CONTEXT=task_performance_str,
						GLOBAL_AGENT_PROFILE=global_profile_str,
					)
				)
			else:
				user_prompts.append(
					self.evolve_mastered_prompt.user_prompt.format(
						MASTERED_TASK=example_str,
						GLOBAL_AGENT_PROFILE=global_profile_str,
					)
				)

		if not user_prompts:
			print("Could not generate any prompts. Skipping task generation.")
			return []

		system_prompt = self._build_system_prompt(self.evolve_mastered_prompt)
		parsed_responses = self._query_and_parse_responses(system_prompt, user_prompts)

		return self._organize_data(
			parsed_responses, parent_sets, example_sets, session_idx, "mastered"
		)

	def _get_valid_parent_statuses(self) -> list[str]:
		"""Returns the list of task statuses that are valid for selecting examples."""
		return [
			"seed",
			"interesting",
			"desc_generated",
			"compile_success",
			"mastered",
			"learnable",
			"unlearnable",
			"A",
			"B",
			"C",
			"D",
			"example",
		]

	def _get_task_performance_str(self, task_id: str) -> str:
		"""Retrieves and formats the performance history for a specific task."""
		try:
			task_data = self.archive.graph.nodes[task_id]
			performance_history = task_data.get("performance_history", [])
			if performance_history:
				task_specific_profile = performance_history[-1]
				return self._format_task_performance_context(task_specific_profile)
			return "No specific performance data found for this task."
		except Exception as e:
			print(f"Warning: Could not retrieve performance history for {task_id}: {e}")
			return f"Error retrieving performance data: {e}"

	def _build_system_prompt(self, prompt_module) -> str:
		"""Builds the system prompt for task generation LLM calls."""
		return prompt_module.system_prompt.format(
			CONSTANTS=CONSTANTS,
			MOBS=MOBS,
			GAME_MECHANICS=GAME_MECHANICS,
			WORLD_GEN=WORLD_GEN,
			API_DOCS=API_DOCS,
		)

	def _query_and_parse_responses(
		self, system_prompt: str, user_prompts: list[str], max_retries: int = 10
	) -> list[dict]:
		"""Queries the LLM and parses responses with a retry loop for failed parses.

		Args:
			system_prompt: The system prompt for the LLM.
			user_prompts: List of user prompts to send to the LLM.
			max_retries: Maximum number of retry attempts for failed parses.

		Returns:
			A list of successfully parsed response dictionaries.
		"""
		responses = self.llm.query(system_prompt, user_prompts)

		final_parsed_responses = [None] * len(responses)
		indices_to_retry = list(range(len(responses)))

		for attempt in range(max_retries):
			for i, original_index in enumerate(indices_to_retry):
				response = responses[i]
				parsed_data = self._parse_generation_response(response)
				if parsed_data.get("description") is not None:
					final_parsed_responses[original_index] = parsed_data

			indices_to_retry = [
				i for i, result in enumerate(final_parsed_responses) if result is None
			]

			if not indices_to_retry:
				print("Successfully parsed all LLM responses.")
				break

			if attempt < max_retries - 1:
				print(
					f"Failed to parse {len(indices_to_retry)} responses. "
					f"Retrying (Attempt {attempt + 2}/{max_retries})..."
				)
				prompts_to_retry = [user_prompts[i] for i in indices_to_retry]
				responses = self.llm.query(system_prompt, prompts_to_retry)
			else:
				print(
					f"Warning: Failed to parse {len(indices_to_retry)} responses "
					f"after {max_retries} attempts."
				)

		return [res for res in final_parsed_responses if res is not None]

	def _format_file_mastered_task(self, example_paths: list[str]) -> str:
		"""Formats a parent task description into a string for the LLM prompt."""
		descriptions = self.archive.get_task_descriptions(example_paths)
		return "\n".join([f"\n{desc}\n" for desc in descriptions.values()])

	def _format_global_agent_profile(self, evaluation_feedback: dict | None) -> str:
		"""Formats the global evaluation metrics into a string for the LLM prompt.

		Args:
			evaluation_feedback: Dictionary of skill metrics from the agent's
				evaluation on the full Craftax game.

		Returns:
			A formatted string describing the agent's skill profile.
		"""
		if not evaluation_feedback:
			return "No overall performance evaluation is available for the agent yet."

		context_str = "This is the agent's *general* skill profile from the full Craftax game:\n"

		# Filter to only show skills with non-zero success rates
		skill_lines = [
			f"- {key}: {value:.2f}%"
			for key, value in evaluation_feedback.items()
			if key.startswith("skill_") and value > 0
		]
		if skill_lines:
			context_str += (
				"Its average performance on key skills was:\n" + "\n".join(skill_lines) + "\n"
			)
		else:
			context_str += "No skills were mastered in the global evaluation.\n"

		return context_str


	def _format_task_performance_context(self, task_profile: dict) -> str:
		"""Formats the task-specific metrics into a string for the LLM prompt."""
		if not task_profile:
			return "No task-specific performance data available."

		context_str = "While training on this task, the agent achieved:\n"

		# Add the main SR for the task's goal (composite success)
		if "sr" in task_profile:
			# 'sr' is a 0-1 ratio, so MULTIPLY by 100
			sr_percent = task_profile["sr"] * 100.0
			context_str += f"- Main Goal Success Rate (SR): {sr_percent:.2f}%\n"

		# Add the SR for ALL individual achievements (goal-related and accidental)
		if "achievement_srs" in task_profile and task_profile["achievement_srs"]:
			context_str += "Detailed Skill SRs (including goal components and accidental skills):\n"
			for key, value in task_profile["achievement_srs"].items():
				# 'value' from achievement_srs is ALREADY 0-100, so DO NOT multiply
				context_str += f"  - {key}: {value:.2f}%\n"
		else:
			context_str += "No detailed skill SRs were recorded.\n"

		return context_str

	def _parse_generation_response(self, response: dict) -> dict:
		"""Parses the LLM's response to extract both the reasoning and the docstring.
		Returns a dictionary with both fields.
		"""
		response_content = response.get("content", "")

		# Default values in case parsing fails
		parsed_data = {"reasoning": None, "description": None}

		# Extract reasoning
		if response_content is None:
			return parsed_data

		reasoning_match = re.search(
			r"<reasoning>\s*(.*?)\s*</reasoning>", response_content, re.DOTALL
		)
		if reasoning_match:
			parsed_data["reasoning"] = reasoning_match.group(1).strip()

		# Extract task description (docstring)
		desc_match = re.search(r"<docstring>\s*(.*?)\s*</docstring>", response_content, re.DOTALL)
		if desc_match:
			parsed_data["description"] = desc_match.group(1).strip()

		return parsed_data

	def _organize_data(
		self,
		parsed_responses: list[dict],
		parent_sets: list[list[str]],
		example_sets: list[list[str]],
		session_idx: int,
		evolution_type: str,
	) -> list[dict]:
		"""Organizes parsed LLM responses into generation result dictionaries.

		Records new tasks in the archive and builds the result structure
		for downstream processing.

		Args:
			parsed_responses: List of parsed LLM response dictionaries.
			parent_sets: List of parent task ID lists for each response.
			example_sets: List of example task ID lists for each response.
			session_idx: Current curriculum session index.
			evolution_type: Type of evolution (e.g., 'mastered').

		Returns:
			A list of generation result dictionaries.
		"""
		generation_results = []
		for i, parsed_data in enumerate(parsed_responses):
			description = parsed_data.get("description")
			if description is None:
				continue  # Skip if the LLM failed to generate a valid docstring

			# Create the new task ID and add it to the archive
			new_task_id = f"task_{self.task_num_counter}"
			parent_tasks = parent_sets[i]
			self.archive.record_new_task(
				child_task=new_task_id,
				parent_tasks=parent_tasks,
				description=description,
				session_id=session_idx,
			)

			# Get the single parent ID
			parent_id = parent_tasks[0] if parent_tasks else None

			# Get the description dictionary and safely access the value
			parent_descriptions = self.archive.get_task_descriptions(parent_tasks)
			parent_description_str = parent_descriptions.get(parent_id, "Parent task not found")

			# Append all relevant data for this new task to our results list
			generation_results.append(
				{
					"generated_task_id": new_task_id,
					"parent_task_id": parent_id,
					"parent_task": parent_description_str,  # Assuming one parent for evolution
					"evolution_type": evolution_type,
					"reasoning": parsed_data.get("reasoning"),
					"docstring": description,
					"examples": example_sets[i],  # Keep track of examples used
				}
			)

			self.task_num_counter += 1

		return generation_results


class EnvGenerator:
	"""Handles the technical process of generating runnable environment code from a
	task description, including a reflection loop to fix compilation errors.
	"""

	def __init__(self, env_generator_llm: LLM, archive: TaskArchive, config):
		"""Initializes the EnvGenerator.

		Args:
		    env_generator_llm: An instance of the LLM class for code generation.
		    archive: An instance of the TaskArchive.
		    config: The Hydra configuration object.

		"""
		self.llm = env_generator_llm
		self.archive = archive
		self.config = config
		self.gen_env_prompt = importlib.import_module(self.config.prompts.env_generation)
		self.craftax_mechanics = importlib.import_module(self.config.prompts.craftax_code).context

		if config.mode != "reward":
			self.wrapper_mechanics = importlib.import_module(
				self.config.prompts.wrapper_mechanics
			).context
		else:
			self.wrapper_mechanics = importlib.import_module(
				self.config.prompts.wrapper_mechanics_r
			).context

	def generate(self, tasks_to_generate: list[dict]) -> dict:
		"""Generates and validates environment files, ensuring compilation for all tasks.
		If a task fails compilation after all reflection attempts, it is re-queued
		for generation from scratch until it succeeds.
		"""
		print(
			f"Generating environment code for {len(tasks_to_generate)} tasks with persistent retries..."
		)

		# Initialize state tracker for each task
		task_states = [
			{
				"task_info": task_info,
				"status": "needs_initial_generation",
				"final_code": None,
				"last_response": None,
				"error_msg": None,
				"reflection_count": 0,
			}
			for task_info in tasks_to_generate
		]

		# Main loop: continues until all tasks compile successfully
		round_num = 0
		while True:
			round_num += 1
			print(f"\n--- Generation/Reflection Round {round_num} ---")

			# Check for completion condition: if all tasks are successful, break the loop.
			if all(s["status"] == "success" for s in task_states):
				print("All tasks have been successfully compiled. Exiting loop.")
				break

			# Separate tasks by their current state
			tasks_for_initial_generation = [
				s for s in task_states if s["status"] == "needs_initial_generation"
			]
			tasks_for_reflection = [
				s for s in task_states if s["status"] == "pending_reflection"
			]

			initial_gen_prompts = []
			if tasks_for_initial_generation:
				print(
					f"Preparing {len(tasks_for_initial_generation)} tasks for initial generation..."
				)
				for state in tasks_for_initial_generation:
					# Reset reflection count for this new attempt
					state["reflection_count"] = 0
					example_str = self._format_code_examples(state["task_info"]["examples"])
					initial_gen_prompts.append(
						self.gen_env_prompt.user_prompt.format(
							CODE_EXAMPLES=example_str,
							TASK_DESCRIPTION=state["task_info"]["description"],
						)
					)

			reflection_prompts = []
			if tasks_for_reflection:
				print(f"Preparing {len(tasks_for_reflection)} tasks for reflection...")
				for state in tasks_for_reflection:
					reflection_prompts.append(
						self._build_reflection_prompt(
							state["last_response"],
							state["error_msg"],
							self._format_code_examples(state["task_info"]["examples"]),
							state["task_info"]["description"],
						)
					)

			system_prompt = self.gen_env_prompt.system_prompt.format(
				CRAFTAX_CODE=self.craftax_mechanics, MINICRAFTAX_CODE=self.wrapper_mechanics
			)

			# Query LLM in parallel for both batches
			new_initial_responses = []
			if initial_gen_prompts:
				new_initial_responses = self.llm.query(system_prompt, initial_gen_prompts)

			new_reflection_responses = []
			if reflection_prompts:
				new_reflection_responses = self.llm.query(system_prompt, reflection_prompts)

			# Update task states with LLM responses
			for i, state in enumerate(tasks_for_initial_generation):
				state["last_response"] = new_initial_responses[i]["content"]
				state["status"] = "pending_validation"

			for i, state in enumerate(tasks_for_reflection):
				state["last_response"] = new_reflection_responses[i]["content"]
				state["reflection_count"] += 1
				state["status"] = "pending_validation"

			# Validate all tasks that have received a new response
			tasks_to_validate = [s for s in task_states if s["status"] == "pending_validation"]
			print(f"Validating {len(tasks_to_validate)} new code attempts...")
			for state in tasks_to_validate:
				code_attempt = self._extract_file(state["last_response"])
				if not code_attempt:
					state["error_msg"] = "Could not extract Python code from the LLM response."
					print(f"Task {state['task_info']['task']}... FAILED (code extraction)")
				else:
					is_correct, error_msg = self.check_compilation(code_attempt)
					if is_correct:
						state["status"] = "success"
						state["final_code"] = code_attempt
						print(f"Task {state['task_info']['task']}... SUCCESS")
					else:
						state["error_msg"] = error_msg
						print(
							f"Task {state['task_info']['task']}... FAILED (compilation error): {error_msg}"
						)

				# Decide the next step for failed tasks
				if state["status"] != "success":
					if state["reflection_count"] >= self.config.num_reflections_max:
						# Ran out of reflection trials, reset for a completely new attempt
						print(
							f"Task {state['task_info']['task']} has failed max reflections. Re-queueing for generation from scratch."
						)
						state["status"] = "needs_initial_generation"
					else:
						# Still have reflection trials left
						state["status"] = "pending_reflection"

		# Final processing and archive update
		generation_results = {}
		for state in task_states:
			task_id = state["task_info"]["task"]
			self.archive.update_node_status(task_id, "compile_success")
			self.archive.update_node_code(task_id, state["final_code"])

			generation_results[task_id] = {
				"compiled": True,
				"code": state["final_code"],
				"error": None,
			}

		print("\nBatch environment generation complete.")
		return generation_results

	def generate_code_only(self, tasks_to_generate: list[dict]) -> dict[str, str | None]:
		"""Runs the LLM query to generate code for a batch of tasks, but does NOT compile.
		This method is safe to run in a background thread.

		Args:
		    tasks_to_generate: A list of dicts, e.g.,
		        [{'task': 'task_123', 'description': '...', 'examples': [...]}]

		Returns:
		    A dictionary mapping task_id to the generated code string (or None if extraction failed).

		"""
		print(f"    WORKER (Thread): Generating code for {len(tasks_to_generate)} tasks...")

		# 1. Prepare a batch of prompts
		user_prompts = []
		code_example_strs = []
		for task_info in tasks_to_generate:
			example_str = self._format_code_examples(task_info["examples"])
			code_example_strs.append(example_str)
			user_prompts.append(
				self.gen_env_prompt.user_prompt.format(
					CODE_EXAMPLES=example_str, TASK_DESCRIPTION=task_info["description"]
				)
			)

		system_prompt = self.gen_env_prompt.system_prompt.format(
			CRAFTAX_CODE=self.craftax_mechanics, MINICRAFTAX_CODE=self.wrapper_mechanics, MOBS=MOBS_CODE,
		)

		# Query the LLM for all prompts in parallel
		responses = self.llm.query(system_prompt, user_prompts)

		# Extract code and map to task_id
		results = {}
		for i, task_info in enumerate(tasks_to_generate):
			task_id = task_info["task"]
			response_content = responses[i].get("content")
			extracted_code = self._extract_file(response_content)
			results[task_id] = extracted_code  # Will be None if extraction failed

		print("    WORKER (Thread): Code generation complete.")
		return results

	def _build_reflection_prompt(
		self, failed_response_content: str, error_msg: str, code_examples_str: str, task_desc: str
	) -> str:
		"""Builds a reflection prompt to help the LLM fix its previous error.

		Args:
			failed_response_content: The LLM's previous response that failed.
			error_msg: The error message from the failed attempt.
			code_examples_str: Formatted code examples for context.
			task_desc: The task description being generated.

		Returns:
			A formatted reflection prompt string.
		"""
		prompt_template = self.gen_env_prompt.user_prompt_reflection_not_compilation_error
		return prompt_template.format(
			PREVIOUS_RESPONSE=failed_response_content, ERROR=error_msg, TASK_DESC=task_desc
		)

	def _format_code_examples(self, example_paths: list[str]) -> str:
		"""Formats the selected code examples into a string for the LLM prompt."""
		codes = self.archive.get_task_codes(example_paths)
		return "\n".join([f"<example>\n{code}\n</example>\n" for code in codes.values()])

	def check_compilation(self, code: str) -> tuple[bool, str]:
		"""Validates code by loading and running a full environment step on CPU.

		This ensures generated code is syntactically correct and produces valid
		JAX-compatible state. Runs strictly on CPU to avoid GPU memory conflicts
		with training.

		Args:
			code: The Python source code to validate.

		Returns:
			A tuple of (success: bool, error_message: str).
		"""
		temp_file = None
		module_name = None

		try:
			# Write code to temporary file
			with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
				f.write(code)
				temp_file = f.name

			# Get CPU device for isolated execution
			try:
				cpu_device = jax.devices("cpu")[0]
			except IndexError:
				cpu_device = jax.local_devices(backend="cpu")[0]

			# Load and validate environment on CPU
			with jax.default_device(cpu_device):
				temp_task = Task(temp_file)
				env = temp_task.env
				module_name = temp_task.task.__module__

				params = env.default_params
				key = jax.random.PRNGKey(0)

				# Define validation kernel that runs reset + step
				def _validate_on_cpu_impl(rng):
					rng, reset_key = jax.random.split(rng)
					obs, state = env.reset(reset_key, params)
					action = env.action_space(params).sample(rng)
					obs, state, reward, done, info = env.step(rng, state, action, params)

					# Validate inventory field types to prevent JAX compilation errors
					for field_name, value in state.inventory.__dict__.items():
						if hasattr(value, "dtype") and value.dtype != jnp.int32:
							raise ValueError(
								f"Inventory field '{field_name}' has type {value.dtype}, expected int32."
							)
					return reward

				_validate_on_cpu = jax.jit(_validate_on_cpu_impl, backend="cpu")
				_ = _validate_on_cpu(key)

			return True, ""

		except Exception as e:
			traceback.print_exc()
			return False, f"Compilation error: {str(e)}"

		finally:
			if temp_file and os.path.exists(temp_file):
				os.unlink(temp_file)
			if module_name and module_name in sys.modules:
				del sys.modules[module_name]

	def _extract_file(self, content: str) -> str | None:
		"""Extracts Python code from an LLM response wrapped in <code> tags.

		Args:
			content: The raw LLM response string.

		Returns:
			The extracted code string, or the original content if no tags found.
		"""
		if not content:
			return None
		code_match = re.search(r"<code>\s*(.*?)\s*</code>", content, re.DOTALL)
		if code_match:
			return code_match.group(1).strip()
		return content


class GenManager:
	"""Main orchestrator for the DiCode evolution pipeline.

	Coordinates task generation, code synthesis, and archive management
	for curriculum learning through LLM-based task evolution.
	"""

	def __init__(self, config):
		"""Initializes the GenManager pipeline.

		Args:
			config: The Hydra configuration object containing all settings.
		"""
		self.config_ = config
		self.config = config.gen_manager

		task_designer = LLM(
			provider=self.config.task_generator.provider,
			base_url=self.config.task_generator.base_url,
			model=self.config.task_generator.model,
			llm_type=self.config.task_generator.llm_type,
			max_tokens=self.config.task_generator.max_tokens,
			temperature=self.config.task_generator.temperature,
			top_p=self.config.task_generator.top_p,
			think=self.config.task_generator.think,
		)
		env_coder = LLM(
			provider=self.config.env_generator.provider,
			base_url=self.config.env_generator.base_url,
			model=self.config.env_generator.model,
			llm_type=self.config.env_generator.llm_type,
			max_tokens=self.config.env_generator.max_tokens,
			temperature=self.config.env_generator.temperature,
			top_p=self.config.env_generator.top_p,
			think=self.config.env_generator.think,
		)

		embedding_model = LLM(
			provider=self.config.embedding_model.provider,
			base_url=self.config.embedding_model.base_url,
			model=self.config.embedding_model.model,
			llm_type=self.config.embedding_model.llm_type,
			embedding_size=self.config.embedding_model.embedding_size,
		)
		self.archive = TaskArchive(self.config)
		self.selector = TaskSelector(self.archive, embedding_model, self.config)
		self.task_generator = TaskGenerator(task_designer, self.archive, self.selector, self.config)
		self.env_generator = EnvGenerator(env_coder, self.archive, self.config)

		self.session_idx = self.archive.get_max_session_idx() + 1

	def evolve_tasks(
		self, dict_of_tasks: dict[str, list[str]], global_agent_profile: dict
	) -> list[dict]:
		"""Orchestrates the full I/O-bound evolution pipeline for one session.

		This method is thread-safe and performs:
		1. Task description generation from mastered tasks (LLM Call 1)
		2. Code generation for new task descriptions (LLM Call 2)
		3. Result merging and preparation for the main thread

		Args:
			dict_of_tasks: Dictionary mapping task categories to task ID lists.
			global_agent_profile: Agent skill metrics from evaluation.

		Returns:
			A list of dictionaries containing generation results.
		"""
		print("    WORKER (Thread): Starting task design...")
		all_generation_results = []

		# --- 1. Evolve from Mastered Tasks (LLM Call 1a) ---

		if not self.config_.ablation:
			if dict_of_tasks.get("mastered"):
				mastered_results = self.task_generator.evolve_mastered(
					self.session_idx, dict_of_tasks["mastered"], global_agent_profile
				)
				all_generation_results.extend(mastered_results)

		else:
			if dict_of_tasks.get("mastered"):
				mastered_results = self.task_generator.evolve_ablation(
					self.session_idx, dict_of_tasks["mastered"], global_agent_profile
				)
				all_generation_results.extend(mastered_results)

		if not all_generation_results:
			print("    WORKER (Thread): No new tasks were designed in this session.")
			return []

		print(
			f"    WORKER (Thread): Task design finished. {len(all_generation_results)} designs created."
		)

		# Prepare for Code Generation
		tasks_to_generate_code_for = [
			{
				"task": result["generated_task_id"],
				"description": result["docstring"],
				"examples": result["examples"],
			}
			for result in all_generation_results
		]

		# Generate Code (LLM Call 2)
		compilation_results = self.env_generator.generate_code_only(tasks_to_generate_code_for)

		# Merge Generation and Compilation Results
		final_results_for_worker = []
		for gen_result in all_generation_results:
			task_id = gen_result["generated_task_id"]
			code_string = compilation_results.get(task_id)

			gen_result["code_string"] = code_string

			if code_string:
				gen_result["compiled"] = None  # To be filled by main thread
				gen_result["code"] = code_string
				gen_result["error"] = None
			else:
				gen_result["compiled"] = False
				gen_result["code"] = ""
				gen_result["error"] = "Failed to extract code from LLM response."

			final_results_for_worker.append(gen_result)

		return final_results_for_worker