"""SPAM galaxy collision simulator (JSPAM), callable from Python."""

import numpy as np

from jspam._spam import simr_interface

N_PARAMS = 22
POINT_MASS = 0
EXTENDED = 1

# How heat is added to the disk velocities. See run_collision.
HEAT_MODES = {"wallin": 0, "centered": 1}

# The unit galaxy's mass model (EXTENDED potential), in the order set_mass_model takes them. The defaults
# are Wallin's values exactly as the Fortran stores them: h_bulge and m_bulge were single-precision
# literals, so they're 0.2 and 0.3333 rounded to single precision.
MASS_MODEL_DEFAULTS = {
    "lnl": 0.001,  # strength of dynamical friction (Coulomb logarithm)
    "r_scale": 10.0,  # halo core radius (rchalo); also sets the tables' radius grid, so it stretches everything
    "m_halo": 5.8,  # halo mass
    "m_bulge": float(np.float32(0.3333)),  # bulge mass
    "h_bulge": float(np.float32(0.2)),  # bulge scale length
    "h_disk": 1.0,  # disk scale length; also the disk's mass
}
# What the mass tables are divided by. "wallin": the fixed 7.13333 (the sum of the default masses), so
# changing a mass changes the galaxy's total mass. "total": this model's actual total, so it's always 1.
NORMALIZATIONS = {"wallin": 0, "total": 1}

# How far back SPAM's start-time search integrates (fixed in getTStart's call), and its step.
SEARCH_WINDOW = 30.0
STEP = 0.001


def run_collision(
    params: np.ndarray,
    n1: int = 1000,
    n2: int = 1000,
    potential: int = EXTENDED,
    seed: int | None = None,
    heat1: float = 0.0,
    heat2: float = 0.0,
    heat_mode: str = "wallin",
    mass_model: dict[str, float] | None = None,
    normalization: str = "wallin",
    t_start: float | None = None,
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
        mass_model: values to change from `MASS_MODEL_DEFAULTS`; the rest keep their defaults. Used
            by the EXTENDED potential. Both galaxies share it, scaled by their masses.
        normalization: "wallin" (the original) or "total". See `NORMALIZATIONS`.
        t_start: when the run starts (SPAM time, now = 0, so negative). None lets SPAM's search choose,
            as in the original (see `timing`).

    Returns:
        x, y, z, vx, vy, vz at the end of the run, shape (n1 + n2 + 1, 6). The last row is the
        secondary galaxy's center.
    """
    params = _check_params(params)
    _set_mass_model(mass_model, normalization)
    _set_start(t_start)
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
    mass_model: dict[str, float] | None = None,
    normalization: str = "wallin",
    t_start: float | None = None,
) -> np.ndarray:
    """Set up one collision and return every particle at the start, before any step.

    Takes the same arguments as `run_collision`. With the same seed, this is exactly the state
    `run_collision` starts stepping from.

    Returns:
        x, y, z, vx, vy, vz at the start time, shape (n1 + n2 + 1, 6). The last row is the
        secondary galaxy's center.
    """
    params = _check_params(params)
    _set_mass_model(mass_model, normalization)
    _set_start(t_start)
    _seed(seed)
    return simr_interface.initial_state(params, potential, n1, n2, heat1, heat2, _heat_mode(heat_mode))


def timing(
    params: np.ndarray,
    potential: int = EXTENDED,
    mass_model: dict[str, float] | None = None,
    normalization: str = "wallin",
    t_start: float | None = None,
) -> dict[str, float | int | bool]:
    """When a run starts, how many steps it takes, and what the start-time search found.

    Only the two galaxy centers are set up; nothing is stepped, so this is fast. Takes the same
    physics arguments as `run_collision` (no particles or seed are needed). With `t_start` given, the
    search doesn't run: "t_start" is that value, and the search's values are 0.

    Returns:
        "t_start": the start time a run uses (SPAM time units, now = 0).
        "n_steps": steps from t_start to the end of a run (1 to 2 steps past 0, as in basic_run).
        "start_fallback": True when t_start is SPAM's -5 fallback, used when the search's start was
            earlier than -12 or within one step of 0.
        "t_start_found": the search's start, before that fallback.
        "t_closest", "closest_distance", "closest_speed": the closest approach between the centers in
            the search's window (back to t = -30).
    """
    params = _check_params(params)
    _set_mass_model(mass_model, normalization)
    _set_start(t_start)
    start, n_steps, fallback, search = simr_interface.get_timing(params, potential)
    return {
        "t_start": float(start),
        "n_steps": int(n_steps),
        "start_fallback": bool(fallback),
        "t_start_found": float(search[0]),
        "t_closest": float(search[3]),
        "closest_distance": float(search[1]),
        "closest_speed": float(search[2]),
    }


def run_orbit(
    params: np.ndarray,
    n_steps: int | None = None,
    potential: int = EXTENDED,
    mass_model: dict[str, float] | None = None,
    normalization: str = "wallin",
    t_start: float | None = None,
) -> np.ndarray:
    """The secondary galaxy's center at every step of a run, with no particles.

    Args:
        params, potential, mass_model, normalization, t_start: as in `run_collision`.
        n_steps: how many steps to take from the start. None takes a full run's (`timing`'s n_steps),
            so the last row is where `run_collision` leaves the secondary's center.

    Returns:
        x, y, z, vx, vy, vz, t at the start and after each step, shape (n_steps + 1, 7).
    """
    params = _check_params(params)
    if n_steps is None:
        n_steps = timing(params, potential, mass_model, normalization, t_start)["n_steps"]
    _set_mass_model(mass_model, normalization)
    _set_start(t_start)
    return simr_interface.run_orbit(params, potential, n_steps)


def search_orbit(
    params: np.ndarray,
    potential: int = EXTENDED,
    mass_model: dict[str, float] | None = None,
    normalization: str = "wallin",
) -> np.ndarray:
    """The path SPAM's start-time search integrates: the secondary's center from now back to -30.

    The two centers are run backward from the model's position now, one step (0.001) at a time, to
    `-SEARCH_WINDOW`. A run is placed by stepping back along this same path, so its start lies on it.

    Returns:
        x, y, z, vx, vy, vz, t, one row per step from t = 0 back to -30, shape (about 30001, 7).
        Velocities are as in forward time.
    """
    params = _check_params(params)
    _set_mass_model(mass_model, normalization)
    nmax = int(SEARCH_WINDOW / STEP) + 3
    path, nrows = simr_interface.search_orbit(params, potential, nmax)
    return path[:nrows].copy()


def mass_profile(mass_model: dict[str, float] | None = None, normalization: str = "wallin") -> dict[str, np.ndarray]:
    """The unit galaxy's tables, as the integrator reads them.

    Args:
        mass_model, normalization: as in `run_collision`.

    Returns:
        Arrays of shape (10000,), one row per table entry: "radius" (SPAM distance units, where the
        integrator picks that row), enclosed mass of "halo", "disk", "bulge" and "total" (for a galaxy
        of mass 1 in the 22 parameters), and "force", the acceleration a particle feels there.
    """
    _set_mass_model(mass_model, normalization)
    nrows = simr_interface.mass_profile_size()
    names = ("radius", "halo", "disk", "bulge", "total", "force")
    return dict(zip(names, simr_interface.mass_profile(nrows)))


def _check_params(params: np.ndarray) -> np.ndarray:
    params = np.asarray(params, dtype=float)
    if params.shape != (N_PARAMS,):
        raise ValueError(f"expected {N_PARAMS} parameters, got shape {params.shape}")
    return params


def _heat_mode(heat_mode: str) -> int:
    if heat_mode not in HEAT_MODES:
        raise ValueError(f"heat_mode must be one of {list(HEAT_MODES)}: {heat_mode!r}")
    return HEAT_MODES[heat_mode]


def _set_mass_model(mass_model: dict[str, float] | None, normalization: str) -> None:
    """Set the mass model in the Fortran. Called before every run, so one run's model can't leak into the next."""
    values = dict(MASS_MODEL_DEFAULTS)
    for name, value in (mass_model or {}).items():
        if name not in values:
            raise ValueError(f"unknown mass model value {name!r}; expected some of {list(values)}")
        values[name] = float(value)
    if normalization not in NORMALIZATIONS:
        raise ValueError(f"normalization must be one of {list(NORMALIZATIONS)}: {normalization!r}")
    # The Fortran takes them by position, in MASS_MODEL_DEFAULTS' order
    ordered = np.array([values[name] for name in MASS_MODEL_DEFAULTS], dtype=float)
    simr_interface.set_mass_model(ordered, NORMALIZATIONS[normalization])


def _set_start(t_start: float | None) -> None:
    """Set the start time in the Fortran, or clear it. Called before every run, so it can't leak."""
    if t_start is None:
        simr_interface.set_start_time(0.0, 0)
    else:
        if not t_start < 0:
            raise ValueError(f"t_start must be before now (negative): {t_start!r}")
        simr_interface.set_start_time(float(t_start), 1)


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
