from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax.training.train_state import TrainState
from dicode.network import ActorCriticRNN, ScannedRNN
from orbax.checkpoint import CheckpointManager, PyTreeCheckpointer
from dicode.ppo_tr import ActorCriticTransformer


def load_full_checkpoint(checkpoint_path, env, env_params, config):
	"""Loads the FULL checkpoint (Weights + Optimizer State + Step Count).
	This ensures global learning rate annealing resumes correctly.
	"""
	print(f"Loading full checkpoint from: {checkpoint_path}")

	# 1. Path Parsing (Same as before)
	ckpt_path_obj = Path(checkpoint_path)
	target_step = None
	root_dir = ckpt_path_obj

	try:
		target_step = int(ckpt_path_obj.name)
		root_dir = ckpt_path_obj.parent
		print(f"Path detected as specific step {target_step}. Root: {root_dir}")
	except ValueError:
		print("Path detected as root directory. Will find latest step.")
		pass

	# 2. Define Network & Global Schedule
	# We must define the optimizer with the GLOBAL schedule here so the restored
	# state is attached to the correct logic.
	rng = jax.random.PRNGKey(0)
	network = ActorCriticTransformer(
		action_dim=env.action_space(env_params).n,
		activation=config.activation,
		encoder_size=config.embed_size,
		hidden_layers=config.hidden_layers,
		num_heads=config.num_heads,
		qkv_features=config.qkv_features,
		num_layers=config.num_layers,
		gating=config.gating,
		gating_bias=config.gating_bias,
	)

	rng, _rng = jax.random.split(rng)
	# Dummy inputs for initialization
	init_obs = jnp.zeros((2, env.observation_space(env_params).shape[0]))
	init_memory = jnp.zeros((2, config.window_mem, config.num_layers, config.embed_size))
	init_mask = jnp.zeros((2, config.num_heads, 1, config.window_mem + 1), dtype=jnp.bool_)
	network_params = network.init(_rng, init_memory, init_obs, init_mask)

	# --- DEFINE GLOBAL SCHEDULE ---
	if config.anneal_lr:
		TOTAL_GLOBAL_UPDATES = config.total_timesteps // config.num_envs // config.num_steps
		anneal_method = getattr(config, "anneal_method", "linear")

		def global_schedule(count):
			current_update_idx = count // (config.num_minibatches * config.update_epochs)
			if anneal_method == "exponential":
				return config.lr * (config.lr_decay_rate**current_update_idx)
			else:
				frac = 1.0 - (current_update_idx / TOTAL_GLOBAL_UPDATES)
				return config.lr * jnp.maximum(0.0, frac)

		tx = optax.chain(
			optax.clip_by_global_norm(config.max_grad_norm),
			optax.adam(learning_rate=global_schedule, eps=1e-5),
		)
	else:
		tx = optax.chain(
			optax.clip_by_global_norm(config.max_grad_norm),
			optax.adam(config.lr, eps=1e-5),
		)

	# Create dummy structure with the CORRECT optimizer definition
	dummy_train_state = TrainState.create(
		apply_fn=network.apply,
		params=network_params,
		tx=tx,
	)

	# 3. Checkpoint Manager & Resolution (Same as before)
	orbax_checkpointer = PyTreeCheckpointer()
	checkpoint_manager = CheckpointManager(root_dir, orbax_checkpointer)

	if target_step is None:
		target_step = checkpoint_manager.latest_step()
		if target_step is None:
			# ... (Keep existing fallback logic if you wish, or just raise error) ...
			if root_dir.exists():
				# Simple fallback to find max integer folder
				steps = [int(d.name) for d in root_dir.iterdir() if d.is_dir() and d.name.isdigit()]
				if steps:
					target_step = max(steps)

		if target_step is None:
			raise ValueError(f"No valid checkpoint steps found in {root_dir}")

	print(f"Restoring step: {target_step}")

	# 4. Restore Full State (Preserving Count & Momentum)
	restored_train_state = None

	# Attempt A: Nested Checkpoint ({"train_state": ...})
	try:
		print("Attempting to restore as NESTED checkpoint...")
		# We pass dummy_train_state as the target structure.
		# Orbax fills .params and .opt_state from disk, but keeps .tx from dummy_train_state.
		target = {"train_state": dummy_train_state}
		restored = checkpoint_manager.restore(target_step, items=target)

		if "train_state" in restored:
			restored_train_state = restored["train_state"]
			print("Success: Restored full NESTED state (Count preserved).")
	except Exception as e:
		print(f"Nested restore failed: {e}")

	# Attempt B: Flat Checkpoint (Direct TrainState)
	if restored_train_state is None:
		try:
			print("Attempting to restore as FLAT checkpoint...")
			restored_train_state = checkpoint_manager.restore(target_step, items=dummy_train_state)
			print("Success: Restored full FLAT state (Count preserved).")
		except Exception as e:
			print(f"Flat restore failed: {e}")

	if restored_train_state is None:
		raise ValueError("Failed to restore checkpoint in either Nested or Flat format.")

	# 5. Return the restored state directly
	# DO NOT create a fresh train state here, or you will lose the count again.
	return restored_train_state

def load_weights_only(checkpoint_path, env, env_params, config, load_opt_state=False):
    """
    Restores checkpoint. 
    If load_opt_state=True, it STRICTLY requires restoring the optimizer state 
    (step count, moments) or it will raise an error.
    """
    print(f"Loading checkpoint from: {checkpoint_path} | load_opt_state={load_opt_state}")

    # --- 1. Path Parsing ---
    ckpt_path_obj = Path(checkpoint_path)
    target_step = None
    root_dir = ckpt_path_obj

    if ckpt_path_obj.name.isdigit():
        target_step = int(ckpt_path_obj.name)
        root_dir = ckpt_path_obj.parent
        print(f"  > Detected specific step: {target_step}")
    else:
        print("  > Path is root directory. Auto-detecting latest step...")

    # --- 2. Initialize Network & Optimizer Definition ---
    # We need the exact structure to match the saved PyTree
    rng = jax.random.PRNGKey(0)
    network = ActorCriticTransformer(
        action_dim=env.action_space(env_params).n,
        activation=config.activation,
        encoder_size=config.embed_size,
        hidden_layers=config.hidden_layers,
        num_heads=config.num_heads,
        qkv_features=config.qkv_features,
        num_layers=config.num_layers,
        gating=config.gating,
        gating_bias=config.gating_bias,
    )

    # Init Params
    rng, _rng = jax.random.split(rng)
    vis_batch_size = 2
    init_obs = jnp.zeros((vis_batch_size, env.observation_space(env_params).shape[0]))
    init_memory = jnp.zeros((vis_batch_size, config.window_mem, config.num_layers, config.embed_size))
    init_mask = jnp.zeros((vis_batch_size, config.num_heads, 1, config.window_mem + 1), dtype=jnp.bool_)
    network_params = network.init(_rng, init_memory, init_obs, init_mask)

    # Reconstruct Optimizer (Must match training.py EXACTLY)
    if config.anneal_lr:
        TOTAL_GLOBAL_UPDATES = (
            (config.total_timesteps // config.num_envs // config.num_steps // config.max_updates_per_session) + 1
        ) * config.max_updates_per_session
        
        def linear_schedule(count):
            frac = 1.0 - (count // (config.num_minibatches * config.update_epochs)) / TOTAL_GLOBAL_UPDATES
            return config.min_lr + (config.lr - config.min_lr) * frac
            
        tx = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )
    else:
        tx = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(config.lr, eps=1e-5),
        )

    # Create Abstract TrainState (The "Template" for restoration)
    dummy_train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)

    # --- 3. Checkpoint Manager Setup ---
    checkpointer = ocp.PyTreeCheckpointer()
    options = ocp.CheckpointManagerOptions(create=False)
    checkpoint_manager = ocp.CheckpointManager(root_dir, checkpointer, options=options)

    if target_step is None:
        target_step = checkpoint_manager.latest_step()
        if target_step is None:
            raise ValueError(f"No checkpoint steps found in {root_dir}")

    print(f"  > Restoring step: {target_step}")

    # --- 4. Restoration Logic ---
    restored_state = None
    
    # Attempt A: Restore as NESTED dict (standard for many setups) -> {'train_state': ...}
    try:
        # We tell Orbax what structure we expect
        target_structure = {"train_state": dummy_train_state}
        restored = checkpoint_manager.restore(target_step, items=target_structure)
        
        if "train_state" in restored:
            restored_state = restored["train_state"]
            print("  > Success: Restored via NESTED structure.")
    except Exception as e:
        # print(f"  > Nested restore attempt skipped: {e}")
        pass

    # Attempt B: Restore as FLAT TrainState (if Attempt A failed)
    if restored_state is None:
        try:
            restored_state = checkpoint_manager.restore(target_step, items=dummy_train_state)
            if hasattr(restored_state, 'params'):
                print("  > Success: Restored via FLAT structure.")
            else:
                restored_state = None # It returned something, but not a valid TrainState
        except Exception as e:
            # print(f"  > Flat restore attempt skipped: {e}")
            pass

    # --- 5. Validation & Return ---
    
    # CASE 1: We wanted the Optimizer State
    if load_opt_state:
        if restored_state is None:
            raise ValueError(
                "CRITICAL: load_opt_state=True, but failed to restore TrainState structure. "
                "Cannot resume optimization schedule."
            )
        
        # Verify Step Count
        try:
            # Extract step count from Optax state
            opt_state = restored_state.opt_state
            # This is a heuristic to find the step count in standard Optax chains
            step_count = jax.tree_util.tree_leaves(opt_state)[0] 
            print(f"  > VERIFICATION: Restored Optimizer Step Count = {step_count}")
            
            if step_count == 0 and target_step > 0:
                 print("  > WARNING: Restored step count is 0, but target_step > 0. Schedule might reset!")
        except Exception:
            print("  > Note: Could not verify internal optimizer step count (structure complex).")

        return restored_state

    # CASE 2: We only wanted Weights (or restore failed but we can salvage weights)
    pretrained_params = None
    if restored_state is not None:
        pretrained_params = restored_state.params
    else:
        # Attempt C: Raw Fallback (Last resort for params only)
        print("  > Fallback: Attempting raw param extraction...")
        try:
            handler = ocp.PyTreeCheckpointHandler()
            # Construct path manually usually: root/step/default
            ckpt_dir = root_dir / str(target_step) / "default"
            full_state = handler.restore(ckpt_dir)
            if "train_state" in full_state:
                pretrained_params = full_state["train_state"]["params"]
            elif "params" in full_state:
                pretrained_params = full_state["params"]
        except Exception as e:
             raise ValueError(f"CRITICAL: Could not recover params even via raw extraction. {e}")

    if pretrained_params is None:
        raise ValueError("Failed to load parameters.")

    print("  > Creating FRESH TrainState with loaded weights (Optimizer Reset).")
    fresh_train_state = TrainState.create(
        apply_fn=network.apply,
        params=pretrained_params,
        tx=tx,
    )
    return fresh_train_state


# def load_weights_only(checkpoint_path, env, env_params, config, load_opt_state=False):
# 	"""Robustly load network weights from either a root dir or step dir, and flat or nested checkpoint."""
# 	print(f"Loading weights from: {checkpoint_path}")

# 	# 1. Path Parsing: Decide if this is a Root Dir or specific Step Dir
# 	ckpt_path_obj = Path(checkpoint_path)
# 	target_step = None
# 	root_dir = ckpt_path_obj

# 	try:
# 		# Check if the path basename is an integer (e.g., .../103450)
# 		target_step = int(ckpt_path_obj.name)
# 		root_dir = ckpt_path_obj.parent
# 		print(f"Path detected as specific step {target_step}. Root: {root_dir}")
# 	except ValueError:
# 		# Not an integer, assume it's the root directory
# 		print("Path detected as root directory. Will find latest step.")
# 		pass

# 	# 2. Initialize Network (Standard / Shared Logic)
# 	# We need the network structure to create the dummy TrainState for restore
# 	rng = jax.random.PRNGKey(0)
# 	network = ActorCriticTransformer(
# 		action_dim=env.action_space(env_params).n,
# 		activation=config.activation,
# 		encoder_size=config.embed_size,
# 		hidden_layers=config.hidden_layers,
# 		num_heads=config.num_heads,
# 		qkv_features=config.qkv_features,
# 		num_layers=config.num_layers,
# 		gating=config.gating,
# 		gating_bias=config.gating_bias,
# 	)

# 	# Create abstract inputs
# 	rng, _rng = jax.random.split(rng)
# 	vis_batch_size = 2
# 	init_obs = jnp.zeros((vis_batch_size, env.observation_space(env_params).shape[0]))
# 	init_memory = jnp.zeros(
# 		(
# 			vis_batch_size,
# 			config.window_mem,
# 			config.num_layers,
# 			config.embed_size,
# 		)
# 	)
# 	init_mask = jnp.zeros(
# 		(vis_batch_size, config.num_heads, 1, config.window_mem + 1),
# 		dtype=jnp.bool_,
# 	)
# 	network_params = network.init(_rng, init_memory, init_obs, init_mask)

# 	# # Create dummy optimizer
# 	# dummy_tx = optax.chain(
# 	# 	optax.clip_by_global_norm(config.max_grad_norm),
# 	# 	optax.adam(config.lr, eps=1e-5),
# 	# )

# 	# # Create dummy TrainState (Target Layout)
# 	# dummy_train_state = TrainState.create(
# 	# 	apply_fn=network.apply,
# 	# 	params=network_params,
# 	# 	tx=dummy_tx,
# 	# )

# 	if config.anneal_lr:
# 		anneal_method = getattr(config, "anneal_method", "linear")
# 		if anneal_method == "exponential":
# 			lr_schedule = optax.exponential_decay(
# 				init_value=config.lr, transition_steps=1, decay_rate=config.lr_decay_rate
# 			)
# 			tx = optax.chain(
# 				optax.clip_by_global_norm(config.max_grad_norm),
# 				optax.adam(learning_rate=lr_schedule, eps=1e-5),
# 			)
# 		else:
# 			TOTAL_GLOBAL_UPDATES = (
# 				(
# 					config.total_timesteps
# 					// config.num_envs
# 					// config.num_steps
# 					// config.max_updates_per_session
# 				)
# 				+ 1
# 			) * config.max_updates_per_session
# 			def linear_schedule(count):
# 				frac = (
# 					1.0 - (count // (config.num_minibatches * config.update_epochs)) / TOTAL_GLOBAL_UPDATES
# 				)
# 				return config.min_lr + (config.lr - config.min_lr) * frac
# 			tx = optax.chain(
# 				optax.clip_by_global_norm(config.max_grad_norm),
# 				optax.adam(learning_rate=linear_schedule, eps=1e-5),
# 			)
# 	else:
# 		tx = optax.chain(
# 			optax.clip_by_global_norm(config.max_grad_norm),
# 			optax.adam(config.lr, eps=1e-5),
# 		)

# 	dummy_train_state = TrainState.create(
# 		apply_fn=network.apply,
# 		params=network_params,
# 		tx=tx, # Use the real tx
# 	)



# 	# 3. Checkpoint Manager
# 	orbax_checkpointer = PyTreeCheckpointer()
# 	checkpoint_manager = CheckpointManager(root_dir, orbax_checkpointer)

# 	# Resolve Step if not specified
# 	if target_step is None:
# 		target_step = checkpoint_manager.latest_step()
# 		if target_step is None:
# 			# Fallback: Manual scan (Legacy dicode logic)
# 			if root_dir.exists():
# 				subdirs = [d for d in root_dir.iterdir() if d.is_dir()]
# 				numeric_steps = []
# 				for d in subdirs:
# 					try:
# 						numeric_steps.append((float(d.name), d.name))
# 					except ValueError:
# 						continue
# 				if numeric_steps:
# 					# Sort by float value, pick the name
# 					_, target_step_name = max(numeric_steps, key=lambda x: x[0])
# 					# We might need to handle float steps specially if Orbax doesn't like them as ints
# 					# For now, let's assume if Orbax failed, we might be in trouble anyway,
# 					# but let's try to trust the manager first.
# 					# If manager failed, maybe it's because they aren't standard integer steps?
# 					# Let's try to rely on what we found.
# 					# Note: 'restore' expects an int step usually, or we pass the directory directly logic?
# 					# Actually Orbax restore takes 'step' as int.
# 					pass

# 			if target_step is None:
# 				raise ValueError(f"No valid checkpoint steps found in {root_dir}")

# 	print(f"Restoring step: {target_step}")

# 	# 4. Robust Restore (Try-Except Block)
# 	# Attempt A: Nested (Standard for 'tr') -> {"train_state": ...}
# 	pretrained_params = None

# 	try:
# 		print("Attempting to restore as NESTED checkpoint...")
# 		target = {"train_state": dummy_train_state}
# 		restored = checkpoint_manager.restore(target_step, items=target)
# 		# Validate
# 		if "train_state" in restored and hasattr(restored["train_state"], "params"):

# 			# --- NEW CHECK ---
# 			if load_opt_state:
# 				print("Success: Loaded NESTED checkpoint with Optimizer State.")
# 				return restored["train_state"]
# 			# -----------------
# 			pretrained_params = restored["train_state"].params
# 			print("Success: Loaded NESTED checkpoint.")
# 			# pretrained_params = restored["train_state"].params
# 			# print("Success: Loaded NESTED checkpoint.")
# 			# # DEBUG: Print shape of first layer kernel if possible
# 			# try:
# 			# 	print(f"[DEBUG] Loaded params type: {type(pretrained_params)}")
# 			# 	# Try to access a known layer if possible, or just print keys
# 			# 	# print(f"[DEBUG] Param keys: {pretrained_params.keys()}")
# 			# except:
# 			# 	pass
# 	except Exception as e_nested:
# 		msg = str(e_nested)
# 		if len(msg) > 200:
# 			msg = msg[:200] + "... [truncated]"
# 		print(f"Nested restore failed ({msg}).")

# 	# Attempt B: Flat (Standard for 'dicode') -> TrainState directly
# 	if pretrained_params is None:
# 		try:
# 			print("Attempting to restore as FLAT checkpoint...")
# 			restored = checkpoint_manager.restore(target_step, items=dummy_train_state)
# 			if hasattr(restored, "params"):
# 				# --- NEW CHECK ---
# 				if load_opt_state:
# 					print("Success: Loaded FLAT checkpoint with Optimizer State.")
# 					return restored
# 				# -----------------
# 				pretrained_params = restored.params
# 				print("Success: Loaded FLAT checkpoint.")
# 		# 	if hasattr(restored, "params"):
# 		# 		pretrained_params = restored.params
# 		# 		print("Success: Loaded FLAT checkpoint.")
# 		except Exception:
# 			print("Flat restore failed.")

# 	# Attempt C: Raw Param Extraction (Last Resort / Old fallback)
# 	if pretrained_params is None:
# 		print("Attempting RAW parameter extraction (last resort)...")
# 		try:
# 			# Construct path manually if needed, or use handler directly
# 			checkpoint_dir = root_dir / str(target_step) / "default"
# 			handler = ocp.PyTreeCheckpointHandler()
# 			full_state = handler.restore(checkpoint_dir)

# 			if "params" in full_state:
# 				pretrained_params = full_state["params"]
# 			elif "train_state" in full_state and "params" in full_state["train_state"]:
# 				pretrained_params = full_state["train_state"]["params"]

# 			print("Success: Loaded parameters via RAW extraction.")
# 		except Exception as e_raw:
# 			raise ValueError(
# 				f"CRITICAL FAILURE: Could not load params using any method. Errors: {e_raw}"
# 			)

# 	if pretrained_params is None:
# 		raise ValueError("Failed to extract parameters from checkpoint.")

# 	# DEBUG: Print shape of first layer kernel (Moved to execute for ANY load method)
# 	try:
# 		print(f"[DEBUG] Loaded params type: {type(pretrained_params)}")

# 		# Handle common Flax structure where everything is under 'params'
# 		params_to_check = pretrained_params
# 		if (
# 			isinstance(pretrained_params, dict)
# 			and "params" in pretrained_params
# 			and len(pretrained_params) == 1
# 		):
# 			print("[DEBUG] Unwrapping outer 'params' key...")
# 			params_to_check = pretrained_params["params"]

# 		# Try to access a known layer to get input shape
# 		if "transformer" in params_to_check and "encoder" in params_to_check["transformer"]:
# 			kernel = params_to_check["transformer"]["encoder"]["kernel"]
# 			print(
# 				f"[DEBUG] Policy Expected Input Shape (from dicode.transformer.encoder): {kernel.shape[0]}"
# 			)
# 		elif "Dense_0" in params_to_check:
# 			kernel = params_to_check["Dense_0"]["kernel"]
# 			print(f"[DEBUG] Policy Expected Input Shape (from Dense_0): {kernel.shape[0]}")
# 		else:
# 			print(f"[DEBUG] Could not auto-detect first layer. Top keys: {params_to_check.keys()}")
# 	except Exception as e:
# 		print(f"[DEBUG] Error inspecting params: {e}")

# 	# 5. Create Fresh Train State with Pretrained Weights
# 	print("Creating fresh optimizer state for curriculum learning...")

# 	fresh_train_state = TrainState.create(
# 		apply_fn=network.apply,
# 		params=pretrained_params,
# 		tx=tx,
# 	)

# 	return fresh_train_state


# def load_weights_only(checkpoint_path, env, env_params, config):
# 	"""Load only network weights for curriculum learning, reset optimizer state."""
# 	rng = jax.random.PRNGKey(0)
# 	# Init network
# 	network = ActorCriticTransformer(
# 		action_dim=env.action_space(env_params).n,
# 		activation=config.activation,
# 		encoder_size=config.embed_size,
# 		hidden_layers=config.hidden_layers,
# 		num_heads=config.num_heads,
# 		qkv_features=config.qkv_features,
# 		num_layers=config.num_layers,
# 		gating=config.gating,
# 		gating_bias=config.gating_bias,
# 	)
#
# 	# Create abstract inputs to init the network
# 	rng, _rng = jax.random.split(rng)
# 	vis_batch_size = 2
# 	# Use batch size > 1 for initialization
# 	init_obs = jnp.zeros((vis_batch_size, env.observation_space(env_params).shape[0]))
# 	init_memory = jnp.zeros(
# 		(
# 			vis_batch_size,
# 			config.window_mem,
# 			config.num_layers,
# 			config.embed_size,
# 		)
# 	)
# 	init_mask = jnp.zeros(
# 		(vis_batch_size, config.num_heads, 1, config.window_mem + 1),
# 		dtype=jnp.bool_,
# 	)
# 	# This is a dummy call to get network_params shape
# 	network_params = network.init(_rng, init_memory, init_obs, init_mask)
#
# 	# Create a dummy optimizer just to get the structure (we won't use its values)
# 	dummy_tx = optax.chain(
# 		optax.clip_by_global_norm(config.max_grad_norm),
# 		optax.adam(config.lr, eps=1e-5),
# 	)
#
# 	# Create dummy train_state for structure matching
# 	dummy_train_state = TrainState.create(
# 		apply_fn=network.apply,
# 		params=network_params,
# 		tx=dummy_tx,
# 	)
#
# 	# Load checkpoint manager
# 	orbax_checkpointer = PyTreeCheckpointer()
# 	checkpoint_manager = CheckpointManager(checkpoint_path, orbax_checkpointer)
#
# 	# Get the latest step
# 	latest_step = checkpoint_manager.latest_step()
# 	if latest_step is None:
# 		# Try to find checkpoints manually
# 		if os.path.exists(checkpoint_path):
# 			subdirs = [
# 				d
# 				for d in os.listdir(checkpoint_path)
# 				if os.path.isdir(os.path.join(checkpoint_path, d))
# 			]
# 			if subdirs:
# 				try:
# 					numeric_steps = []
# 					for subdir in subdirs:
# 						try:
# 							step_num = float(subdir)
# 							numeric_steps.append((step_num, subdir))
# 						except ValueError:
# 							continue
#
# 					if numeric_steps:
# 						latest_step_num, latest_step_dir = max(numeric_steps, key=lambda x: x[0])
# 						latest_step = latest_step_dir
# 						print(
# 							f"Found checkpoint at step {latest_step_num} (directory: {latest_step_dir})"
# 						)
# 					else:
# 						raise ValueError(f"No valid checkpoint steps found in {checkpoint_path}")
# 				except Exception as e:
# 					raise ValueError(f"Error parsing checkpoint steps in {checkpoint_path}: {e}")
# 			else:
# 				raise ValueError(f"No checkpoint directories found in {checkpoint_path}")
# 		else:
# 			raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
#
# 	# Restore checkpoint - only extract params, ignore optimizer state mismatch
# 	try:
# 		loaded_state = checkpoint_manager.restore(latest_step, dummy_train_state)
# 		pretrained_params = loaded_state.params
# 	except ValueError as e:
# 		# If there's a structure mismatch in opt_state, try to load just params directly
# 		print(f"Standard restore failed ({e}), attempting direct parameter extraction...")
# 		# Use lower-level API to extract just the params
# 		from pathlib import Path
#
# 		import orbax.checkpoint as ocp
#
# 		checkpoint_dir = Path(checkpoint_path) / str(latest_step) / "default"
# 		handler = ocp.PyTreeCheckpointHandler()
#
# 		# Create a target structure with just params
# 		restore_args = ocp.checkpoint_utils.construct_restore_args(dummy_train_state)
# 		# Try to restore with structure that only cares about params
# 		try:
# 			# Restore everything but we'll only use params
# 			full_state = handler.restore(checkpoint_dir)
# 			pretrained_params = full_state["params"]
# 		except:
# 			raise ValueError(f"Could not load parameters from checkpoint at {checkpoint_path}")
#
# 	print("Loaded pretrained weights, creating fresh optimizer state for curriculum learning")
#
# 	# Create fresh optimizer for new task
#
# 	if config.anneal_lr:
# 		if config.anneal_method == "exponential":
# 			# Use exponential decay which is better for variable-length sessions
# 			lr_schedule = optax.exponential_decay(
# 				init_value=config.lr, transition_steps=1, decay_rate=config.lr_decay_rate
# 			)
# 			tx = optax.chain(
# 				optax.clip_by_global_norm(config.max_grad_norm),
# 				optax.adam(learning_rate=lr_schedule, eps=1e-5),
# 			)
# 		else:  # Default to linear schedule
# 			NUM_UPDATES = config.total_timesteps // config.num_steps // config.num_envs
#
# 			def linear_schedule(count):
# 				frac = (
# 					1.0 - (count // (config.num_minibatches * config.update_epochs)) / NUM_UPDATES
# 				)
# 				return config.lr * frac
#
# 			tx = optax.chain(
# 				optax.clip_by_global_norm(config.max_grad_norm),
# 				optax.adam(learning_rate=linear_schedule, eps=1e-5),
# 			)
# 	else:
# 		tx = optax.chain(
# 			optax.clip_by_global_norm(config.max_grad_norm),
# 			optax.adam(config.lr, eps=1e-5),
# 		)
#
# 	# Create fresh train_state with pretrained weights but new optimizer
# 	fresh_train_state = TrainState.create(
# 		apply_fn=network.apply,
# 		params=pretrained_params,  # Use pretrained weights
# 		tx=tx,  # Fresh optimizer state
# 	)
#
# 	return fresh_train_state


# def load_weights_only(checkpoint_path, env, env_params, config):
# 	"""Load only network weights for curriculum learning, reset optimizer state."""

# 	# Initialize network architecture
# 	network = ActorCriticRNN(env.action_space(env_params).n, config=config)
# 	rng = jax.random.PRNGKey(0)
# 	init_x = (
# 		jnp.zeros((1, config.num_envs, *env.observation_space(env_params).shape)),
# 		jnp.zeros((1, config.num_envs)),
# 	)
# 	init_hstate = ScannedRNN.initialize_carry(config.num_envs, config.layer_size)
# 	network_params = network.init(rng, init_hstate, init_x)

# 	# Create a dummy optimizer just to get the structure (we won't use its values)
# 	dummy_tx = optax.chain(
# 		optax.clip_by_global_norm(config.max_grad_norm),
# 		optax.adam(config.lr, eps=1e-5),
# 	)

# 	# Create dummy train_state for structure matching
# 	dummy_train_state = TrainState.create(
# 		apply_fn=network.apply,
# 		params=network_params,
# 		tx=dummy_tx,
# 	)

# 	# Load checkpoint manager
# 	orbax_checkpointer = PyTreeCheckpointer()
# 	checkpoint_manager = CheckpointManager(checkpoint_path, orbax_checkpointer)

# 	# Get the latest step
# 	latest_step = checkpoint_manager.latest_step()
# 	if latest_step is None:
# 		# Try to find checkpoints manually
# 		if os.path.exists(checkpoint_path):
# 			subdirs = [d for d in os.listdir(checkpoint_path) if os.path.isdir(os.path.join(checkpoint_path, d))]
# 			if subdirs:
# 				try:
# 					numeric_steps = []
# 					for subdir in subdirs:
# 						try:
# 							step_num = float(subdir)
# 							numeric_steps.append((step_num, subdir))
# 						except ValueError:
# 							continue

# 					if numeric_steps:
# 						latest_step_num, latest_step_dir = max(numeric_steps, key=lambda x: x[0])
# 						latest_step = latest_step_dir
# 						print(f"Found checkpoint at step {latest_step_num} (directory: {latest_step_dir})")
# 					else:
# 						raise ValueError(f"No valid checkpoint steps found in {checkpoint_path}")
# 				except Exception as e:
# 					raise ValueError(f"Error parsing checkpoint steps in {checkpoint_path}: {e}")
# 			else:
# 				raise ValueError(f"No checkpoint directories found in {checkpoint_path}")
# 		else:
# 			raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")

# 	# Restore checkpoint - only extract params, ignore optimizer state mismatch
# 	try:
# 		loaded_state = checkpoint_manager.restore(latest_step, dummy_train_state)
# 		pretrained_params = loaded_state.params
# 	except ValueError as e:
# 		# If there's a structure mismatch in opt_state, try to load just params directly
# 		print(f"Standard restore failed ({e}), attempting direct parameter extraction...")
# 		# Use lower-level API to extract just the params
# 		import orbax.checkpoint as ocp
# 		from pathlib import Path

# 		checkpoint_dir = Path(checkpoint_path) / str(latest_step) / "default"
# 		handler = ocp.PyTreeCheckpointHandler()

# 		# Create a target structure with just params
# 		restore_args = ocp.checkpoint_utils.construct_restore_args(dummy_train_state)
# 		# Try to restore with structure that only cares about params
# 		try:
# 			# Restore everything but we'll only use params
# 			full_state = handler.restore(checkpoint_dir)
# 			pretrained_params = full_state['params']
# 		except:
# 			raise ValueError(f"Could not load parameters from checkpoint at {checkpoint_path}")

# 	print("Loaded pretrained weights, creating fresh optimizer state for curriculum learning")

# 	# Create fresh optimizer for new task

# 	if config.anneal_lr:
# 		if config.anneal_method == "exponential":
# 			# Use exponential decay which is better for variable-length sessions
# 			lr_schedule = optax.exponential_decay(
# 				init_value=config.lr,
# 				transition_steps=1,
# 				decay_rate=config.lr_decay_rate
# 			)
# 			tx = optax.chain(
# 				optax.clip_by_global_norm(config.max_grad_norm),
# 				optax.adam(learning_rate=lr_schedule, eps=1e-5),
# 			)
# 		else: # Default to linear schedule
# 			NUM_UPDATES = config.total_timesteps // config.num_steps // config.num_envs
# 			def linear_schedule(count):
# 				frac = 1.0 - (count // (config.num_minibatches * config.update_epochs)) / NUM_UPDATES
# 				return config.lr * frac
# 			tx = optax.chain(
# 				optax.clip_by_global_norm(config.max_grad_norm),
# 				optax.adam(learning_rate=linear_schedule, eps=1e-5),
# 			)
# 	else:
# 		tx = optax.chain(
# 			optax.clip_by_global_norm(config.max_grad_norm),
# 			optax.adam(config.lr, eps=1e-5),
# 		)

# 	# Create fresh train_state with pretrained weights but new optimizer
# 	fresh_train_state = TrainState.create(
# 		apply_fn=network.apply,
# 		params=pretrained_params,  # Use pretrained weights
# 		tx=tx,  # Fresh optimizer state
# 	)

# 	return fresh_train_state


def load_checkpoint(checkpoint_path, env, env_params, config):
	"""Load a checkpoint and return the train_state."""
	# Initialize network architecture (needed for train_state creation)
	network = ActorCriticRNN(env.action_space(env_params).n, config=config)
	rng = jax.random.PRNGKey(0)  # Dummy key for initialization
	init_x = (
		jnp.zeros((1, config.num_envs, *env.observation_space(env_params).shape)),
		jnp.zeros((1, config.num_envs)),
	)
	init_hstate = ScannedRNN.initialize_carry(config.num_envs, config.layer_size)
	network_params = network.init(rng, init_hstate, init_x)

	if config.anneal_lr:
		if config.anneal_method == "exponential":
			# Re-create the exponential schedule used during the original training
			lr_schedule = optax.exponential_decay(
				init_value=config.lr, transition_steps=1, decay_rate=config.lr_decay_rate
			)
			tx = optax.chain(
				optax.clip_by_global_norm(config.max_grad_norm),
				optax.adam(learning_rate=lr_schedule, eps=1e-5),
			)
		else:  # Default to the linear schedule
			NUM_UPDATES = config.total_timesteps // config.num_steps // config.num_envs

			def linear_schedule(count):
				frac = (
					1.0 - (count // (config.num_minibatches * config.update_epochs)) / NUM_UPDATES
				)
				return config.lr * frac

			tx = optax.chain(
				optax.clip_by_global_norm(config.max_grad_norm),
				optax.adam(learning_rate=linear_schedule, eps=1e-5),
			)
	else:
		tx = optax.chain(
			optax.clip_by_global_norm(config.max_grad_norm),
			optax.adam(config.lr, eps=1e-5),
		)

	# Create a dummy train_state for structure matching during restore
	dummy_train_state = TrainState.create(
		apply_fn=network.apply,
		params=network_params,
		tx=tx,
	)

	# Load checkpoint
	orbax_checkpointer = PyTreeCheckpointer()
	checkpoint_manager = CheckpointManager(checkpoint_path, orbax_checkpointer)

	# Get the latest step (or specify a step number)
	latest_step = checkpoint_manager.latest_step()
	if latest_step is None:
		# Try to find checkpoints manually in case they're saved with float step names
		import os

		if os.path.exists(checkpoint_path):
			subdirs = [
				d
				for d in os.listdir(checkpoint_path)
				if os.path.isdir(os.path.join(checkpoint_path, d))
			]
			if subdirs:
				# Try to convert directory names to numbers and find the latest
				try:
					numeric_steps = []
					for subdir in subdirs:
						try:
							# Handle both int and float step names
							step_num = float(subdir)
							numeric_steps.append((step_num, subdir))
						except ValueError:
							continue

					if numeric_steps:
						# Use the highest step number
						latest_step_num, latest_step_dir = max(numeric_steps, key=lambda x: x[0])
						latest_step = latest_step_dir  # Use the actual directory name
						print(
							f"Found checkpoint at step {latest_step_num} (directory: {latest_step_dir})"
						)
					else:
						raise ValueError(f"No valid checkpoint steps found in {checkpoint_path}")
				except Exception as e:
					raise ValueError(f"Error parsing checkpoint steps in {checkpoint_path}: {e}")
			else:
				raise ValueError(f"No checkpoint directories found in {checkpoint_path}")
		else:
			raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")

	# Restore the train_state using the actual directory name
	loaded_train_state = checkpoint_manager.restore(latest_step, dummy_train_state)

	return loaded_train_state
