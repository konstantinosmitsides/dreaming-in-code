# --- Standard Library ---
import concurrent.futures
import traceback

# --- Third-Party ---
import jax
import numpy as np
from flax.training.train_state import TrainState
from omegaconf import DictConfig
from dicode.evaluation import evaluate_new_tasks
from dicode.logging_utils import log_wandb_table_append
from dicode.runtime_analysis import tracker
from dicode.scoring import calculate_scores_from_snapshot
from dicode.selection import select_tasks_for_evolution

# --- Local Modules ---
from dicode.dreaming.gen_manager import GenManager, TaskArchive


def _find_worst_active_task(archive: TaskArchive) -> tuple[str | None, float]:
	"""Iterates through the graph to find the active task
	with the lowest priority_score.
	"""
	worst_id = None
	worst_score = np.inf

	# We lock the graph to safely iterate
	with archive._lock:
		# Create a list to avoid iterating while holding the lock if not needed
		# (Though in this simple case, it's fine)
		graph_nodes = list(archive.graph.nodes(data=True))

	for node_id, data in graph_nodes:
		if data.get("is_active"):
			score = data.get("priority_score", 0.0)
			if score < worst_score:
				worst_score = score
				worst_id = node_id

	if worst_id is None:
		return None, np.inf  # No active tasks found

	return worst_id, worst_score


def attempt_to_activate_task(
	gen_manager: GenManager, new_task_id: str, new_task_score: float, config: DictConfig
):
	"""Implements the "compare-and-swap" logic for inserting a new task
	into the active set.
	"""
	# 1. Get current state from config and archive
	active_count = gen_manager.archive.active_task_count
	capacity = config.dicode_manager.active_task_capacity

	score_to_beat = -np.inf
	worst_task_id = None

	# 2. Find the score to beat
	if active_count < capacity:
		# Case 1: Active set is not full. Compare against the minimum entry threshold.
		score_to_beat = config.dicode_manager.min_entry_score_threshold
		print(
			f"    - Active set not full ({active_count}/{capacity}). Comparing to threshold: {score_to_beat:.4f}"
		)
	else:
		# Case 2: Active set is full. Find and compare against the worst active task.
		print(f"    - Active set full ({active_count}/{capacity}). Finding worst active task...")
		worst_task_id, worst_score = _find_worst_active_task(gen_manager.archive)
		score_to_beat = worst_score
		print(f"    - Worst active task is '{worst_task_id}' with score: {worst_score:.4f}")

	# 3. Compare and swap
	if new_task_score >= score_to_beat:
		print(
			f"    - SUCCESS: New task score ({new_task_score:.4f}) > score_to_beat ({score_to_beat:.4f})."
		)

		if worst_task_id is not None:
			# Evict the worst task if the buffer was full
			print(f"    - Evicting task '{worst_task_id}'...")
			gen_manager.archive.set_task_active_status(worst_task_id, False)

		print(f"    - Activating new task '{new_task_id}'...")
		gen_manager.archive.set_task_active_status(new_task_id, True)
		gen_manager.archive.update_node_status(new_task_id, "activated")

	else:
		gen_manager.archive.update_node_status(new_task_id, "not_good_enough")
		print(
			f"    - REJECT (Active Set): New task score ({new_task_score:.4f}) <= score_to_beat ({score_to_beat:.4f})."
		)


def run_session_validation(
	config: DictConfig,
	rng: jax.Array,
	rl_train_state: TrainState,
	gen_manager: GenManager,
	evolve_future: concurrent.futures.Future | None,
	current_session_idx: int,
) -> tuple:
	"""Checks for results from the evolution worker and runs JAX-based validation.

	This function performs the "join" part of the async evolution:
	1. Checks if the `evolve_future` (from the *previous* session) is done.
	2. If so, retrieves results and runs JAX-based compilation.
	3. Runs JAX-based evaluation (`evaluate_new_tasks`) on compiled tasks.
	4. Applies learnability thresholds to archive or reject tasks.
	5. Prepares the `generation_table` for W&B logging.
	"""
	print("--- [Main Thread] Checking for evolution worker results... ---")
	activated_count = 0
	rejected_count = 0
	generation_table = None

	if evolve_future is not None and evolve_future.done():
		print("  - Worker finished. Retrieving results...")
		try:
			# worker_results is the list[dict] from gen_manager.evolve_tasks
			worker_results = evolve_future.result()
			evolve_future = None  # Clear the future

			if worker_results:
				# This new function does the GPU work (check_compilation, evaluate_new_tasks)
				rng, validation_rng = jax.random.split(rng)

				validation_metrics = run_validation_on_main_thread(
					config,
					validation_rng,
					rl_train_state,
					gen_manager,
					worker_results,
				)

				activated_count = validation_metrics.get("activated_count", 0)
				rejected_count = validation_metrics.get("rejected_count", 0)

				if config.use_wandb and worker_results:
					print("Logging LLM generation results to a W&B Table...")

					columns = [
						"session_idx",
						"evolution_type",
						"parent_task_id",
						"parent_task",
						"generated_task_id",
						"compilation_status",
						"LLM_reasoning",
						"generated_docstring",
						"final_code",
					]

					# 1. Create list of new rows
					new_generation_rows = []
					for result in worker_results:
						new_generation_rows.append(
							[
								current_session_idx,
								result.get("evolution_type"),
								result.get("parent_task_id"),
								result.get("parent_task"),
								result.get("generated_task_id"),
								"✅ Success" if result.get("compiled") else "❌ Fail",
								result.get("reasoning"),
								result.get("docstring"),
								result.get("code"),
							]
						)

					# 2. Use the new helper function
					generation_table = log_wandb_table_append(
						table_key="curriculum/generation_table",
						columns=columns,
						new_data_rows=new_generation_rows,
					)

			else:
				print("  - Worker returned no new tasks.")

		except Exception as e:
			print(f"  - ERROR: Evolution worker failed with exception: {e}")
			traceback.print_exc()  # Print the full stack trace
			evolve_future = None  # Clear the failed future

	elif evolve_future is not None:
		print("  - Evolution worker is still running in the background.")
	else:
		print("  - No evolution job pending.")

	return rng, evolve_future, activated_count, rejected_count, generation_table


def dispatch_evolution_worker(
	executor: concurrent.futures.ThreadPoolExecutor,
	evolve_future: concurrent.futures.Future | None,
	gen_manager: GenManager,
	config: DictConfig,
	evaluation_metrics: dict,
) -> concurrent.futures.Future | None:
	"""Dispatches a new evolution worker thread if one is not already running.

	This function performs the "dispatch" part of the async evolution:
	1. Checks if `evolve_future` is None (meaning the main thread is ready).
	2. If so, selects eligible tasks for evolution.
	3. Submits a new I/O-bound job (`gen_manager.evolve_tasks`) to the executor.
	4. Returns the new `Future` object tracking this job.
	"""
	print("--- [Main Thread] Checking to dispatch evolution worker... ---")
	if evolve_future is None:
		print("  - No worker running. Selecting tasks and dispatching new job...")

		# 1. Select tasks (This is main-thread, but fast CPU-work)
		tasks_to_evolve_ids = select_tasks_for_evolution(
			gen_manager.archive, gen_manager.session_idx - 1, config.dicode_manager.evolution_max_tasks_k
		)

		if tasks_to_evolve_ids:
			# 2. Submit the I/O-bound job to the thread pool
			# We call `evolve_tasks`, which does LLM calls but NO JAX.

			def _wrapped_evolve_tasks(tasks, metrics, session_idx):
				tracker.start_timer("Evolution (LLM)")
				res = gen_manager.evolve_tasks(tasks, metrics)
				tracker.stop_timer("Evolution (LLM)", session_idx)
				return res

			evolve_future = executor.submit(
				_wrapped_evolve_tasks,  # This is the thread-safe worker function
				{"mastered": tasks_to_evolve_ids, "unlearnable": []},  # Mimic old API
				evaluation_metrics,  # Pass current eval metrics
				gen_manager.session_idx,  # Pass session idx for logging
			)
			print(
				f"  - Dispatched evolution job for {len(tasks_to_evolve_ids)} tasks to background thread."
			)
		else:
			print("  - No eligible tasks to evolve this session.")
	else:
		print("  - Worker job is already running. Skipping new job dispatch.")

	return evolve_future


def run_validation_on_main_thread(
	config: DictConfig,
	rng: jax.Array,
	train_state: TrainState,
	gen_manager: GenManager,
	worker_results: list[dict],
) -> dict:
	"""Runs the JAX-dependent validation and evaluation on the main thread.
	This function performs all GPU-bound work for new tasks.
	"""
	activated_count = 0
	rejected_count = 0
	tasks_to_validate_ids = []

	if not worker_results:
		print("  - Worker returned no results to validate.")
		return {"archived_count": 0, "rejected_count": 0}

	print(f"  - [Main Thread] Validating {len(worker_results)} new task designs...")

	for res in worker_results:
		task_id = res.get("generated_task_id")
		code_string = res.get("code_string")  # The raw code from the worker
		reasoning = res.get("reasoning")  # Get the reasoning string from the worker result

		if not task_id or not code_string:
			print("  - Skipping result, missing task_id or code_string.")
			rejected_count += 1
			if task_id:
				gen_manager.archive.update_node_status(task_id, "compile_fail_no_code")
			continue

		if reasoning:
			gen_manager.archive.update_node_reasoning(task_id, reasoning)

		# 1. Run JAX-based compilation check ON THE MAIN THREAD
		# This is the "Compile (GPU)" step from your original pipeline
		tracker.start_timer("Compilation (GPU)")
		is_correct, error_msg = gen_manager.env_generator.check_compilation(code_string)
		tracker.stop_timer("Compilation (GPU)", gen_manager.session_idx)

		if is_correct:
			print(f"  - Task {task_id} compiled successfully.")
			gen_manager.archive.update_node_code(task_id, code_string)
			gen_manager.archive.update_node_status(task_id, "compile_success")
			tasks_to_validate_ids.append(task_id)
			res["compiled"] = True
		else:
			print(f"  - Task {task_id} FAILED compilation: {error_msg}")
			gen_manager.archive.update_node_status(task_id, "compile_fail")
			# You can optionally remove the node to save space
			# gen_manager.archive.remove_node(task_id)
			rejected_count += 1

			# Log the failed code and error to the W&B table
			res["compiled"] = False
			res["error"] = error_msg
			# (You will need to adapt your W&B logging to handle this list)

	# 2. Run JAX-based evaluation for all compiled tasks in a batch
	# This is the "Validate (GPU)" step from your original pipeline
	if tasks_to_validate_ids:
		print(f"  - [Main Thread] Evaluating {len(tasks_to_validate_ids)} newly compiled tasks...")
		rng, eval_new_rng = jax.random.split(rng)
		tracker.start_timer("Evaluation (Evolved Tasks)")
		raw_eval_data = evaluate_new_tasks(
			config,
			eval_new_rng,
			train_state,
			tasks_to_validate_ids,
			gen_manager.archive,
			gen_manager.selector.embedding_model,  # Pass the embedding model
		)
		tracker.stop_timer("Evaluation (Evolved Tasks)", gen_manager.session_idx)

		scoring_window_data = raw_eval_data.get("scoring_window_data")
		task_achievement_mask = raw_eval_data.get("task_achievement_mask")
		task_completed_mask = raw_eval_data.get("task_completed_mask")
		num_tasks = len(tasks_to_validate_ids)

		if scoring_window_data is None or num_tasks == 0:
			print("  - [Main Thread] Evaluation returned no data. Skipping thresholding.")
			# We must still reject the tasks that were sent for validation
			for task_id in tasks_to_validate_ids:
				gen_manager.archive.update_node_status(task_id, "rejected_eval_fail")
				rejected_count += 1
			return {"activated_count": activated_count, "rejected_count": rejected_count}

		print("  - [Main Thread] Calling Smart Calculator to score new tasks...")
		# all_task_metrics has integer keys: {"0": {...}, "1": {...}}
		all_task_metrics = calculate_scores_from_snapshot(
			scoring_window_data,
			num_tasks,
			task_achievement_mask,
			task_completed_mask,
			config,  # Pass the full config
		)

		evolved_task_results = {}
		for i, task_id in enumerate(tasks_to_validate_ids):
			metrics = all_task_metrics.get(str(i))
			if metrics:
				evolved_task_results[task_id] = metrics
			else:
				print(
					f"  - Warning: No metrics found from calculator for task {task_id} (index {i})"
				)
				# Give it a default "fail" metric
				evolved_task_results[task_id] = {"sr": -1.0, "priority_score": -np.inf}

		# 3. Apply validation rules (from your original STEP 6)
		print("  - [Main Thread] Applying learnability thresholds...")
		# --- NEW LOOP ---
		for task_id, metrics in evolved_task_results.items():
			# --- NEW: Extract metrics ---
			sr = metrics.get("sr", -1.0)
			# This is the new score (e.g., PVL) from the calculator
			initial_priority_score = metrics.get("priority_score", 0.0)

			# --- OLD CALC (still used for thresholding) ---
			clipped_sr = np.clip(sr, 0.0, 1.0) if sr >= 0 else 0.0
			# This is the simple sr*(1-sr)
			learnability_score = clipped_sr * (1.0 - clipped_sr)

			if sr < 0:
				print(f"    - Task {task_id}: SR={sr:.3f} (No eps finished). Status -> rejected")
				gen_manager.archive.update_node_status(task_id, "rejected_no_episodes")
				# gen_manager.archive.remove_node(task_id)
				rejected_count += 1
			else:
				print(
					f"    - Task {task_id}: SR={sr:.3f}, PriorityScore={initial_priority_score:.4f}. Considering for activation..."
				)

				# 1. Store the scores
				gen_manager.archive.update_node_learnability(task_id, learnability_score)
				gen_manager.archive.update_node_priority_score(task_id, initial_priority_score)

				# 2. Immediately try to activate. This function now holds
				#    all the logic (including the min_entry_score_threshold check).
				attempt_to_activate_task(gen_manager, task_id, initial_priority_score, config)

				# 3. Check the result and count.
				if gen_manager.archive.is_active(task_id):
					print(f"    - Task {task_id} was ACTIVATED.")
					activated_count += 1
				else:
					# The function already set the status to "not_good_enough"
					print(f"    - Task {task_id} was REJECTED from active set.")
					rejected_count += 1
	else:
		print("  - [Main Thread] No new tasks passed compilation.")

	return {"activated_count": activated_count, "rejected_count": rejected_count}
