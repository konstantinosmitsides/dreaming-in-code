"""Paper baselines: PPO-GTrXL-based PLR, DR and SFL, derived from NCC-UED (see NOTICE).

Compatibility shim
------------------
`jaxued` (0.0.1, the version on PyPI) calls `jax.tree_map` throughout. That alias was removed
in JAX 0.6.0 in favour of `jax.tree_util.tree_map`, so importing the baselines against the JAX
this repository installs raises AttributeError deep inside `jaxued.level_sampler`.

Restoring the alias here is deliberate. The alternative is pinning the whole project to
jax<0.6, which would hold DiCode itself back for the sake of one dependency. The two functions
are the same function -- `jax.tree_map` was a pure re-export -- so this changes no behaviour.

This runs before any submodule imports `jaxued`, because Python executes a package's __init__
before its submodules.
"""

import jax as _jax
import jax.tree_util as _jax_tree_util

if not hasattr(_jax, "tree_map"):
    _jax.tree_map = _jax_tree_util.tree_map
