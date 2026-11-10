"""SPAM galaxy collision simulator (JSPAM), callable from Python."""

import numpy as np

from jspam._spam import simr_interface

N_PARAMS = 22
POINT_MASS = 0
EXTENDED = 1

# How heat is added to the disk velocities. See run_collision.
HEAT_MODES = {"wallin": 0, "centered": 1}


def run_collision(
    params: np.ndarray,
    n1: int = 1000,
    n2: int = 1000,
    potential: int = EXTENDED,
    seed: int | None = None,
    heat1: float = 0.0,
    heat2: float = 0.0,
    heat_mode: str = "wallin",
) -> np.ndarray:
    """Run one collision and return the final state of every particle.

    Args:
        params: the 22 collision parameters, in SPAM's model-string order, shape (22,).
        n1: number of particles in the primary galaxy's disk.
        n2: number of particles in the secondary galaxy's disk.
        potential: POINT_MASS or EXTENDED (the halo/disk/bulge potential, NBI in the Fortran).
        seed: sets where the disk particles start. The same seed gives the same particles. None
            seeds from the operating system, so every run differs (the original behavior).
        heat1, heat2: random motion added to each disk's particle velocities. The default, 0, is
            the original's.
        heat_mode: "wallin" (the original) adds a value in [0, heat) to each of vx, vy and vz, which
            is always positive, so a heated disk drifts toward +x, +y, +z. "centered" adds a value in
            [-heat/2, heat/2): the same spread, with no drift.

    Returns:
        x, y, z, vx, vy, vz at the end of the run, shape (n1 + n2 + 1, 6). The last row is the
        secondary galaxy's center.
    """
    params = _check_params(params)
    _seed(seed)
    return simr_interface.run_collision(params, potential, n1, n2, heat1, heat2, _heat_mode(heat_mode))


def initial_state(
    params: np.ndarray,
    n1: int = 1000,
    n2: int = 1000,
    potential: int = EXTENDED,
    seed: int | None = None,
    heat1: float = 0.0,
    heat2: float = 0.0,
    heat_mode: str = "wallin",
) -> np.ndarray:
    """Set up one collision and return every particle at the start, before any step.

    Takes the same arguments as `run_collision`. With the same seed, this is exactly the state
    `run_collision` starts stepping from.

    Returns:
        x, y, z, vx, vy, vz at the start time, shape (n1 + n2 + 1, 6). The last row is the
        secondary galaxy's center.
    """
    params = _check_params(params)
    _seed(seed)
    return simr_interface.initial_state(params, potential, n1, n2, heat1, heat2, _heat_mode(heat_mode))


def _check_params(params: np.ndarray) -> np.ndarray:
    params = np.asarray(params, dtype=float)
    if params.shape != (N_PARAMS,):
        raise ValueError(f"expected {N_PARAMS} parameters, got shape {params.shape}")
    return params


def _heat_mode(heat_mode: str) -> int:
    if heat_mode not in HEAT_MODES:
        raise ValueError(f"heat_mode must be one of {list(HEAT_MODES)}: {heat_mode!r}")
    return HEAT_MODES[heat_mode]


def _seed(seed: int | None) -> None:
    if seed is None:
        simr_interface.seed_from_system()
    else:
        simr_interface.set_seed(_seed_values(seed))


def _seed_values(seed: int) -> np.ndarray:
    """Expand one integer into the seed array the Fortran generator needs.

    `SeedSequence` turns any integer, even 0 or 1, into well-mixed values, so nearby seeds give
    unrelated particles.
    """
    size = simr_interface.seed_size()
    state = np.random.SeedSequence(seed).generate_state(size, dtype=np.uint32)
    return state.view(np.int32)
