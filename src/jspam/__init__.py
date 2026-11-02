"""SPAM galaxy collision simulator (JSPAM), callable from Python."""

import numpy as np

from jspam._spam import simr_interface

N_PARAMS = 22
POINT_MASS = 0
EXTENDED = 1


def run_collision(
    params: np.ndarray, n1: int = 1000, n2: int = 1000, potential: int = EXTENDED
) -> np.ndarray:
    """Run one collision and return the final state of every particle.

    Args:
        params: the 22 collision parameters, in SPAM's model-string order, shape (22,).
        n1: number of particles in the primary galaxy's disk.
        n2: number of particles in the secondary galaxy's disk.
        potential: POINT_MASS or EXTENDED (the halo/disk/bulge potential, NBI in the Fortran).

    Returns:
        x, y, z, vx, vy, vz at the end of the run, shape (n1 + n2 + 1, 6). The last row is the
        secondary galaxy's center. Disk particles start from unseeded random positions.
    """
    params = np.asarray(params, dtype=float)
    if params.shape != (N_PARAMS,):
        raise ValueError(f"expected {N_PARAMS} parameters, got shape {params.shape}")
    return simr_interface.run_collision(params, potential, n1, n2)
