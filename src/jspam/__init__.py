"""SPAM galaxy collision simulator (JSPAM), callable from Python."""

import numpy as np

from jspam._spam import simr_interface

N_PARAMS = 22
POINT_MASS = 0
EXTENDED = 1


def run_collision(
    params: np.ndarray,
    n1: int = 1000,
    n2: int = 1000,
    potential: int = EXTENDED,
    seed: int | None = None,
) -> np.ndarray:
    """Run one collision and return the final state of every particle.

    Args:
        params: the 22 collision parameters, in SPAM's model-string order, shape (22,).
        n1: number of particles in the primary galaxy's disk.
        n2: number of particles in the secondary galaxy's disk.
        potential: POINT_MASS or EXTENDED (the halo/disk/bulge potential, NBI in the Fortran).
        seed: sets where the disk particles start. The same seed gives the same particles. None
            seeds from the operating system, so every run differs (the original behavior).

    Returns:
        x, y, z, vx, vy, vz at the end of the run, shape (n1 + n2 + 1, 6). The last row is the
        secondary galaxy's center.
    """
    params = np.asarray(params, dtype=float)
    if params.shape != (N_PARAMS,):
        raise ValueError(f"expected {N_PARAMS} parameters, got shape {params.shape}")
    if seed is None:
        simr_interface.seed_from_system()
    else:
        simr_interface.set_seed(_seed_values(seed))
    return simr_interface.run_collision(params, potential, n1, n2)


def _seed_values(seed: int) -> np.ndarray:
    """Expand one integer into the seed array the Fortran generator needs.

    `SeedSequence` turns any integer, even 0 or 1, into well-mixed values, so nearby seeds give
    unrelated particles.
    """
    size = simr_interface.seed_size()
    state = np.random.SeedSequence(seed).generate_state(size, dtype=np.uint32)
    return state.view(np.int32)
