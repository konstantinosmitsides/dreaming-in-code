# --- Standard Library ---
import types

# --- Third-Party ---
import numpy as np
from craftax.craftax.constants import Achievement

# --- Local Modules ---
from dicode.dreaming.gen_manager import TaskArchive

# 2. Pre-calculate vector size for efficiency
# We find the highest ID in the Enum and add 1 to handle the 0-index.
# For your Enum, max value is 66, so size is 67.
MAX_ACHIEVEMENT_ID = max(ach.value for ach in Achievement)
EMBEDDING_SIZE = MAX_ACHIEVEMENT_ID + 1


def get_achievement_multi_hot(relevant_achievements: list[Achievement]) -> np.ndarray:
	"""Converts a list of Achievement enums into a fixed-size multi-hot vector.

	Args:
	    relevant_achievements: List of Achievement enum objects.

	Returns:
	    np.ndarray: A float32 array of size (67,) where 1.0 indicates presence.

	"""
	# Initialize zero vector
	embedding = np.zeros(EMBEDDING_SIZE, dtype=np.float32)

	# Set active indices to 1.0
	for ach in relevant_achievements:
		embedding[ach.value] = 1.0

	return embedding


def load_tasks_from_env_codes(archive: TaskArchive, tasks: list[str]) -> tuple[list, list]:
	"""Dynamically loads task Env classes from code strings stored in the TaskArchive.

	This avoids the need for separate .py files for each generated task, reading
	the code directly from the graph node attributes.

	Args:
	    archive: The TaskArchive instance.
	    task_dirs: A list of task directory paths (node IDs) to load.

	Returns:
	    A list of the loaded Env classes.

	"""
	task_classes = []
	successful_task_ids = []

	# 1. Get all code strings from the archive in one batch
	code_strings_dict = archive.get_task_codes(tasks)

	for task, code_string in code_strings_dict.items():
		if not code_string:
			print(f"Warning: No code found for task {task}. Skipping.")
			continue

		try:
			# 2. Create a new, temporary "virtual" module in memory
			# We use a unique name to avoid conflicts.
			task_module = types.ModuleType(task)

			# 3. Execute the code string within the namespace of our new module
			exec(code_string, task_module.__dict__)

			# 4. Extract the 'Env' class that was just defined in the module
			if hasattr(task_module, "Env"):
				task_classes.append(task_module.Env)
				successful_task_ids.append(task)
			else:
				print(f"Warning: No 'Env' class found in code for task {task}. Skipping.")

		except Exception as e:
			print(f"Error dynamically loading code for task {task}: {e}")

	return task_classes, successful_task_ids


def get_active_tasks_for_session(
	archive: TaskArchive,
	current_session_idx: int,
	max_active_tasks: int,
	cooldown_period: int,
) -> list[str]:
	"""Selects tasks for the session, capping the total number at max_active_tasks.
	It prioritizes learnable/new tasks, then fills remaining slots with a random
	sample of cooled-down unlearnable tasks.
	"""
	# 1. Get the high-priority "core" tasks (learnable and new)
	previous_session_idx = current_session_idx - 1
	core_tasks = {
		node_id
		for node_id, data in archive.graph.nodes(data=True)
		# if (
		#     data.get("status") == "learnable" and not data.get("re_evaluated") or
		#     data.get("session_created") == previous_session_idx
		# )
		if data.get("session_created") == previous_session_idx
	}

	# Add seed tasks on the first run
	if current_session_idx == 1:
		core_tasks.update(
			{
				node_id
				for node_id, data in archive.graph.nodes(data=True)
				if data.get("status") == "seed"
			}
		)

	# 2. Calculate how many slots are left to fill
	num_slots_to_fill = max(0, max_active_tasks - len(core_tasks))

	if num_slots_to_fill == 0:
		print(
			f"Cap of {max_active_tasks} reached with core tasks. No unlearnable tasks will be added."
		)
		return list(core_tasks)

	# 3. Get all eligible unlearnable tasks and calculate their learnability
	eligible_candidates = []
	for node_id, data in archive.graph.nodes(data=True):
		if node_id in core_tasks:
			continue
		# Check status and cooldown period first
		if data.get("status") in ["A", "B", "C", "D"] and data.get("session_created", 0) <= (
			current_session_idx - cooldown_period
		):
			history = data.get("performance_history", [])
			if history:
				latest_sr = history[-1].get("sr", 0.0)
				# Only consider tasks with a valid, positive SR
				if latest_sr > 0:
					# Calculate learnability: peaks at SR=0.5
					learnability = (1.0 - latest_sr) * latest_sr + 1e-6
					eligible_candidates.append((node_id, learnability))

	# 4. Fill the slots with a weighted sample based on learnability
	tasks_to_add = set()
	if eligible_candidates:
		num_to_sample = min(num_slots_to_fill, len(eligible_candidates))

		# Unzip the candidates into separate lists for task IDs and their scores
		task_ids, learnability_scores = zip(*eligible_candidates)

		# Convert scores to a probability distribution
		total_learnability = sum(learnability_scores)
		probabilities = [score / total_learnability for score in learnability_scores]

		# Sample using numpy's weighted random choice (without replacement)
		chosen_tasks = np.random.choice(
			a=task_ids, size=num_to_sample, replace=False, p=probabilities
		)
		tasks_to_add = set(chosen_tasks)
		print(f"Filling {len(tasks_to_add)} slots with tasks sampled by learnability.")

	# 5. Combine, flag, and return the final set
	final_active_tasks = core_tasks.union(tasks_to_add)
	for task_id in tasks_to_add:
		archive.mark_as_re_evaluated(task_id)

	print(f"Selected {len(final_active_tasks)} active tasks for session {current_session_idx}.")
	return list(final_active_tasks)


def update_archive_statuses(archive: TaskArchive, categorized_tasks: dict):
	"""Updates the status of all task nodes in the archive based on the
	categorized performance data.

	Args:
	    archive: The TaskArchive instance to be updated.
	    categorized_tasks: A dictionary with keys "mastered", "learnable",
	                       and "unlearnable", each containing a list of task IDs.

	"""
	print("Updating task statuses in the archive...")

	# Iterate through each category and its list of tasks
	for status, task_list in categorized_tasks.items():
		# For each task in the list, call the archive's update method
		for task_id in task_list:
			archive.update_node_status(task_id, status)

	print("Archive statuses updated.")


def categorize_active_tasks(
	performance_data: dict,
) -> dict:
	"""Categorizes tasks into 'A', 'B', 'C', and 'D' bins
	based on their final Success Rate (SR).

	Args:
	    performance_data: A dictionary mapping task IDs to their metrics,
	                      e.g., {"task_15": {"sr": 0.95, "lp": 0.01}}.
	    lp_threshold: The LP below which a task is considered to have stalled.

	Returns:
	    A dictionary with keys "A", "B", "C", "D"
	    each containing a list of task IDs.

	"""
	# 1. Initialize the dictionary with empty lists for each category
	categorized_tasks = {
		"A": [],
		"B": [],
		"C": [],
		"D": [],
	}

	# 2. Iterate through each task and its performance
	for task_id, metrics in performance_data.items():
		sr = metrics.get("sr", -1.0)
		lp = metrics.get("lp", 0.0)

		# 3. Apply the criteria to categorize the task
		if sr >= 0.75:
			categorized_tasks["A"].append(task_id)
		elif 0.5 <= sr < 0.75:
			categorized_tasks["B"].append(task_id)
		elif 0.25 <= sr < 0.5:
			categorized_tasks["C"].append(task_id)
		else:
			categorized_tasks["D"].append(task_id)

	return categorized_tasks


if __name__ == "__main__":
	ach_list = [
		Achievement.COLLECT_WOOD,  # ID 0
		Achievement.PLACE_TABLE,  # ID 1
		Achievement.MAKE_WOOD_PICKAXE,  # ID 5
	]
	ach_list_2 = [
		Achievement.MAKE_WOOD_PICKAXE,  # ID 5
		Achievement.COLLECT_WOOD,  # ID 0
		Achievement.PLACE_TABLE,  # ID 1
	]
	print(len(get_achievement_multi_hot(ach_list)))
	print(len(get_achievement_multi_hot(ach_list_2)))
	print(get_achievement_multi_hot(ach_list_2) == get_achievement_multi_hot(ach_list))
