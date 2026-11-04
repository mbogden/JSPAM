"""SPAM galaxy collision simulator (JSPAM), callable from Python."""

from dataclasses import dataclass

import numpy as np

from jspam._spam import simr_interface

N_PARAMS = 22
POINT_MASS = 0
EXTENDED = 1

NORMALIZATIONS = {"wallin": 0, "total": 1}
HEAT_MODES = {"wallin": 0, "centered": 1}

# SPAM's fixed inner disk radius (rin1, rin2 in parameters_module).
DISK_RIN = 0.05


@dataclass(frozen=True)
class MassModel:
    """The galaxy mass model and dynamical friction. Defaults are SPAM's own.

    Attributes:
        lnl: strength of dynamical friction.
        r_scale: radial scale of the halo (`rchalo` in the Fortran).
        m_halo: halo mass.
        m_bulge: bulge mass.
        h_bulge: bulge scale length.
        h_disk: disk scale length. The disk's mass grows with it.
        normalization: "wallin" divides every table by a fixed 7.13333, the total at the default
            values, so other values change the galaxy's total mass. "total" divides by this model's
            actual total mass, so they change only how the mass is distributed.
    """

    lnl: float = 0.001
    r_scale: float = 10.0
    m_halo: float = 5.8
    m_bulge: float = 0.3333
    h_bulge: float = 0.2
    h_disk: float = 1.0
    normalization: str = "wallin"

    def fortran_args(self) -> dict:
        """The keyword arguments the Fortran routines take for this model."""
        if self.normalization not in NORMALIZATIONS:
            raise ValueError(f"normalization must be one of {list(NORMALIZATIONS)}: {self.normalization!r}")
        values = (self.lnl, self.r_scale, self.m_halo, self.m_bulge, self.h_bulge, self.h_disk)
        if not np.isfinite(values).all() or self.lnl < 0 or min(values[1:]) <= 0:
            raise ValueError(f"mass model values must be finite and positive (lnl may be 0): {self}")
        return {
            "lnl_in": self.lnl,
            "r_scale": self.r_scale,
            "m_halo": self.m_halo,
            "m_bulge": self.m_bulge,
            "h_bulge": self.h_bulge,
            "h_disk": self.h_disk,
            "normalization": NORMALIZATIONS[self.normalization],
        }


@dataclass(frozen=True)
class Timing:
    """When a run starts and normally ends.

    Attributes:
        t_start: start time in SPAM units. t = 0 is the moment the 22 parameters describe.
        n_steps: steps to the normal end, which is 1 to 2 steps past t = 0.
        start_fallback: True if SPAM's estimate was rejected (earlier than -12, or within one step of
            0) and the run was moved to start at t = -5.
    """

    t_start: float
    n_steps: int
    start_fallback: bool


def _check_params(params: np.ndarray) -> np.ndarray:
    params = np.asarray(params, dtype=float)
    if params.shape != (N_PARAMS,):
        raise ValueError(f"expected {N_PARAMS} parameters, got shape {params.shape}")
    if not np.isfinite(params).all():
        raise ValueError(f"parameters must be finite: {params}")
    if not params[:3].any() or not params[3:6].any():
        raise ValueError("the secondary's position and velocity can't be zero: the forces would be NaN")
    if params[6] <= 0 or params[7] <= 0:
        raise ValueError(f"masses must be positive: {params[6:8]}")
    if params[8] <= DISK_RIN or params[9] <= DISK_RIN:
        raise ValueError(f"disk radii must be larger than {DISK_RIN}: {params[8:10]}")
    return params


def get_timing(params: np.ndarray, mass_model: MassModel = MassModel(), potential: int = EXTENDED) -> Timing:
    """Find when a run starts and normally ends, without integrating it."""
    t_start, n_steps, fallback = simr_interface.get_timing(
        _check_params(params), potential, **mass_model.fortran_args()
    )
    return Timing(float(t_start), int(n_steps), bool(fallback))


def run_orbit(
    params: np.ndarray,
    mass_model: MassModel = MassModel(),
    potential: int = EXTENDED,
    n_steps: int | None = None,
) -> np.ndarray:
    """Follow the secondary galaxy's center. Deterministic, so it takes no seed.

    Args:
        params: the 22 collision parameters, in SPAM's model-string order, shape (22,).
        mass_model: the galaxy mass model.
        potential: POINT_MASS or EXTENDED (the halo/disk/bulge potential, NBI in the Fortran).
        n_steps: steps to take. Default: the run's normal end.

    Returns:
        x, y, z, vx, vy, vz, t at the start and after each step, shape (n_steps + 1, 7).
    """
    params = _check_params(params)
    if n_steps is None:
        n_steps = get_timing(params, mass_model, potential).n_steps
    if n_steps < 0:
        raise ValueError(f"n_steps can't be negative: {n_steps}")
    return simr_interface.run_orbit(params, potential, n_steps=n_steps, **mass_model.fortran_args())


def _seed_values(seed: int) -> np.ndarray:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer: {seed!r}")
    # SeedSequence spreads small seeds like 1 and 2 across all the integers gfortran's generator needs.
    state = np.random.SeedSequence(int(seed)).generate_state(simr_interface.random_seed_size())
    return state.astype(np.int32)


def run_particles(
    params: np.ndarray,
    *,
    seed: int,
    n1: int = 1000,
    n2: int = 1000,
    mass_model: MassModel = MassModel(),
    potential: int = EXTENDED,
    heat1: float = 0.0,
    heat2: float = 0.0,
    heat_mode: str = "wallin",
    end_steps: list[int] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one collision and save the particles at one or more steps.

    The same inputs and seed give identical numbers from the same build on the same machine. Other
    builds or CPUs can differ in the last digits.

    Args:
        params: the 22 collision parameters, in SPAM's model-string order, shape (22,).
        seed: sets where the disk particles start. Required, so no run is unseeded by accident.
        n1: number of particles in the primary galaxy's disk.
        n2: number of particles in the secondary galaxy's disk.
        mass_model: the galaxy mass model.
        potential: POINT_MASS or EXTENDED (the halo/disk/bulge potential, NBI in the Fortran).
        heat1: random motion added to the primary's particles.
        heat2: random motion added to the secondary's particles.
        heat_mode: "wallin" adds a value in [0, heat) to each velocity component, which also drifts
            the disk. "centered" adds a value in [-heat/2, heat/2): the same spread with no drift.
        end_steps: steps to save at, strictly ascending; 0 is the initial state. Default: the run's
            normal end.

    Returns:
        snapshots: x, y, z, vx, vy, vz of every particle at each step, shape
            (n_ends, n1 + n2 + 1, 6). The last row of each is the secondary galaxy's center.
        steps: the steps saved, shape (n_ends,).
        times: the time of each in SPAM units, shape (n_ends,).
    """
    params = _check_params(params)
    if n1 < 0 or n2 < 0:
        raise ValueError(f"particle counts can't be negative: {n1}, {n2}")
    if heat1 < 0 or heat2 < 0:
        raise ValueError(f"heat can't be negative: {heat1}, {heat2}")
    if heat_mode not in HEAT_MODES:
        raise ValueError(f"heat_mode must be one of {list(HEAT_MODES)}: {heat_mode!r}")
    seed_values = _seed_values(seed)

    if end_steps is None:
        end_steps = [get_timing(params, mass_model, potential).n_steps]
    steps = np.asarray(end_steps)
    if steps.ndim != 1 or steps.size == 0 or not np.issubdtype(steps.dtype, np.integer):
        raise ValueError(f"end_steps must be a non-empty list of integers: {end_steps}")
    if steps[0] < 0 or (np.diff(steps) <= 0).any():
        raise ValueError(f"end_steps must be strictly ascending and not negative: {end_steps}")

    snapshots, times = simr_interface.run_particles(
        params,
        potential,
        npts1=n1,
        npts2=n2,
        heat_1=heat1,
        heat_2=heat2,
        heat_mode=HEAT_MODES[heat_mode],
        seed_size=seed_values.size,
        seed_values=seed_values,
        n_ends=steps.size,
        end_steps=steps,
        **mass_model.fortran_args(),
    )
    return np.moveaxis(snapshots, 2, 0), steps.astype(int), times


def get_built_mass_model() -> dict:
    """The mass model values Fortran last built, for checking they reached the right variables."""
    values = simr_interface.get_built_mass_model()
    keys = ("lnl", "r_scale", "m_halo", "m_bulge", "h_bulge", "h_disk", "normalization")
    return dict(zip(keys, (float(v) if i < 6 else int(v) for i, v in enumerate(values))))
