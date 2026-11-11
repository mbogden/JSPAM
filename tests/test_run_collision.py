"""Checks that the Python interface gives the same answer as the original Fortran program."""

import numpy as np
import pytest

import jspam

# Model SDSS 587722984435351614 from archive/data.txt (the first 22 values).
PARAMS = np.array(
    [
        -9.93853, -4.5805, 15.43348, -1.54025, -2.97899, 4.69234, 34.41671, 29.49607, 2.97913, 4.2501, 301.0,
        35.5, 310.80745, 321.98757, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]
)  # fmt: skip

# Last row of a.101 from the original `basic_run "<model string>"`, built with -O3.
EXPECTED_LAST_ROW = [
    -0.99400701e01, -0.45834789e01, 0.15438172e02, -0.15399158e01, -0.29788359e01, 0.46918210e01,
]  # fmt: skip


def test_matches_original_program():
    final = jspam.run_collision(PARAMS)
    assert final.shape == (2001, 6)
    assert np.isfinite(final).all()
    np.testing.assert_allclose(final[-1], EXPECTED_LAST_ROW, rtol=1e-5, atol=1e-6)


def test_second_run_gives_same_orbit():
    first = jspam.run_collision(PARAMS, n1=10, n2=5)
    second = jspam.run_collision(PARAMS, n1=10, n2=5)
    np.testing.assert_allclose(first[-1], second[-1], rtol=1e-12)


def test_particle_count_does_not_change_orbit():
    final = jspam.run_collision(PARAMS, n1=10, n2=5)
    assert final.shape == (16, 6)
    np.testing.assert_allclose(final[-1], EXPECTED_LAST_ROW, rtol=1e-5, atol=1e-6)


def test_same_seed_gives_same_particles():
    first = jspam.run_collision(PARAMS, n1=50, n2=50, seed=7)
    second = jspam.run_collision(PARAMS, n1=50, n2=50, seed=7)
    np.testing.assert_array_equal(first, second)


def test_different_seeds_give_different_particles():
    first = jspam.run_collision(PARAMS, n1=50, n2=50, seed=7)
    second = jspam.run_collision(PARAMS, n1=50, n2=50, seed=8)
    assert not np.allclose(first[:-1], second[:-1])
    np.testing.assert_allclose(first[-1], second[-1], rtol=1e-12)  # the orbit has no random numbers


def test_no_seed_differs_each_run():
    seeded = jspam.run_collision(PARAMS, n1=50, n2=50, seed=7)
    first = jspam.run_collision(PARAMS, n1=50, n2=50)
    second = jspam.run_collision(PARAMS, n1=50, n2=50)
    assert not np.allclose(first[:-1], second[:-1])
    assert not np.allclose(first[:-1], seeded[:-1])  # an earlier seed doesn't carry over


def test_initial_state_then_run_in_one_process():
    start = jspam.initial_state(PARAMS, n1=10, n2=5, seed=3)
    final = jspam.run_collision(PARAMS, n1=10, n2=5, seed=3)
    assert start.shape == final.shape == (16, 6)
    assert not np.allclose(start, final)  # the run moved the particles


def test_heat_only_changes_starting_velocities():
    """With one seed, heat changes nothing but the disk velocities, by the kick SPAM adds."""
    heat = 0.4
    cold = jspam.initial_state(PARAMS, n1=2000, n2=2000, seed=5)
    hot = jspam.initial_state(PARAMS, n1=2000, n2=2000, seed=5, heat1=heat, heat2=heat)
    np.testing.assert_array_equal(hot[:, :3], cold[:, :3])  # positions
    np.testing.assert_allclose(hot[-1], cold[-1], rtol=1e-12)  # secondary's center
    kick = hot[:-1, 3:] - cold[:-1, 3:]
    # Wallin's heat: randm() * heat, so each component is in [0, heat) and the mean is heat / 2
    assert kick.min() >= -1e-12
    assert kick.max() < heat + 1e-12
    np.testing.assert_allclose(kick.mean(axis=0), heat / 2, rtol=0.05)


def test_centered_heat_has_no_drift():
    """Centered heat: same spread as Wallin's, but each kick is in [-heat/2, heat/2), with mean 0."""
    heat = 0.4
    cold = jspam.initial_state(PARAMS, n1=2000, n2=2000, seed=5)
    hot = jspam.initial_state(PARAMS, n1=2000, n2=2000, seed=5, heat1=heat, heat2=heat, heat_mode="centered")
    wallin = jspam.initial_state(PARAMS, n1=2000, n2=2000, seed=5, heat1=heat, heat2=heat)
    np.testing.assert_array_equal(hot[:, :3], cold[:, :3])  # positions
    kick = hot[:-1, 3:] - cold[:-1, 3:]
    assert kick.min() >= -heat / 2 - 1e-12
    assert kick.max() < heat / 2 + 1e-12
    np.testing.assert_allclose(kick.mean(axis=0), 0.0, atol=0.02 * heat)
    # The same random numbers, shifted by heat / 2: the two modes differ only by the drift
    np.testing.assert_allclose(wallin[:-1, 3:] - hot[:-1, 3:], heat / 2, rtol=1e-9)


def test_rejects_unknown_heat_mode():
    with pytest.raises(ValueError):
        jspam.run_collision(PARAMS, n1=10, n2=5, heat_mode="hot")


# --- Mass model ---------------------------------------------------------------------------------------

WALLIN_NORM = 7.13333  # Wallin's fixed normalization: the sum of the default masses

# A model from archive/data.txt (line 3) whose secondary passes close enough for dynamical friction
# (lnl) to change the orbit. In the SDSS model above it never gets close, so lnl has no effect there.
CLOSE_PARAMS = np.array(
    [
        1.08119, -1.66798, -2.06826, -0.18438, -0.39693, -0.02919, 0.51654, 0.36192, 0.94043, 0.43877, 12.97415,
        135.11902, 97.54035, 305.01225, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]
)  # fmt: skip


def test_explicit_defaults_match_no_mass_model():
    default = jspam.run_collision(PARAMS, n1=10, n2=5, seed=1)
    explicit = jspam.run_collision(
        PARAMS, n1=10, n2=5, seed=1, mass_model=jspam.MASS_MODEL_DEFAULTS, normalization="wallin"
    )
    np.testing.assert_array_equal(default, explicit)


def test_wallin_normalization_total_is_not_one():
    """The true total at the defaults is about 7.101, so dividing by 7.13333 leaves 0.9955."""
    profile = jspam.mass_profile()
    np.testing.assert_allclose(profile["total"][-1], 0.99549, rtol=1e-4)


def test_total_normalization_is_one_whatever_the_masses():
    for mass_model in [{}, {"m_halo": 11.6}, {"m_bulge": 1.0, "h_disk": 2.0}]:
        profile = jspam.mass_profile(mass_model, normalization="total")
        np.testing.assert_allclose(profile["total"][-1], 1.0, rtol=1e-12)


def test_each_mass_value_reaches_its_own_component():
    """Guards against a swap: each value changes the part of the galaxy it names, and only that part."""
    base = jspam.mass_profile()
    # Masses, with Wallin's fixed divisor so each shows up directly
    bulge = jspam.mass_profile({"m_bulge": 0.6})
    np.testing.assert_allclose(bulge["bulge"][-1], 0.6 / WALLIN_NORM, rtol=1e-6)
    np.testing.assert_array_equal(bulge["halo"], base["halo"])
    np.testing.assert_array_equal(bulge["disk"], base["disk"])
    halo = jspam.mass_profile({"m_halo": 11.6})
    np.testing.assert_allclose(halo["halo"], 2 * base["halo"], rtol=1e-12)
    np.testing.assert_array_equal(halo["bulge"], base["bulge"])
    disk = jspam.mass_profile({"h_disk": 2.0})  # the disk's mass equals its scale length
    np.testing.assert_allclose(disk["disk"][-1], 2.0 / WALLIN_NORM, rtol=1e-2)
    np.testing.assert_array_equal(disk["halo"], base["halo"])
    # Shapes: h_bulge spreads the bulge without changing its mass
    wide_bulge = jspam.mass_profile({"h_bulge": 0.5})
    np.testing.assert_allclose(wide_bulge["bulge"][-1], base["bulge"][-1], rtol=1e-6)
    assert wide_bulge["bulge"][100] < base["bulge"][100]  # less of it inside r = 0.1
    np.testing.assert_array_equal(wide_bulge["halo"], base["halo"])
    # r_scale is the halo's core radius, but the whole table's radius grid is built from it too, so it
    # stretches every component (Wallin's design), while the integrator still reads rows at fixed distances
    stretched = jspam.mass_profile({"r_scale": 5.0})
    for component in ("halo", "disk", "bulge"):
        assert not np.allclose(stretched[component], base[component])


def test_lnl_changes_a_close_encounter():
    base = jspam.run_collision(CLOSE_PARAMS, n1=5, n2=5, seed=1, mass_model={"lnl": 0.0})
    friction = jspam.run_collision(CLOSE_PARAMS, n1=5, n2=5, seed=1, mass_model={"lnl": 0.5})
    assert np.abs(base[-1, :3] - friction[-1, :3]).max() > 1e-3


def test_mass_model_does_not_leak_into_the_next_run():
    first = jspam.run_collision(PARAMS, n1=10, n2=5, seed=1)
    jspam.run_collision(PARAMS, n1=10, n2=5, seed=1, mass_model={"m_halo": 11.6}, normalization="total")
    again = jspam.run_collision(PARAMS, n1=10, n2=5, seed=1)
    np.testing.assert_array_equal(first, again)


def test_rejects_unknown_mass_model_values():
    with pytest.raises(ValueError):
        jspam.mass_profile({"mhalo": 5.8})
    with pytest.raises(ValueError):
        jspam.mass_profile(normalization="unit")


def test_rejects_wrong_number_of_parameters():
    with pytest.raises(ValueError):
        jspam.run_collision(PARAMS[:5])


# --- Timing and orbits -------------------------------------------------------------------------------


def test_timing_matches_a_run():
    timing = jspam.timing(PARAMS)
    assert timing["t_start"] < 0
    assert timing["n_steps"] == int(-timing["t_start"] / 0.001) + 2
    assert not timing["start_fallback"]
    assert timing["t_start"] == timing["t_start_found"]
    assert timing["t_start"] < timing["t_closest"] < 0
    assert timing["closest_distance"] > 0 and timing["closest_speed"] > 0
    start = jspam.initial_state(PARAMS, n1=5, n2=5, seed=1)
    orbit = jspam.run_orbit(PARAMS, n_steps=0)
    np.testing.assert_array_equal(orbit[0, :6], start[-1])  # the same starting center
    np.testing.assert_allclose(orbit[0, 6], timing["t_start"])


def test_orbit_ends_where_a_run_leaves_the_secondary():
    orbit = jspam.run_orbit(PARAMS)
    final = jspam.run_collision(PARAMS, n1=10, n2=5, seed=1)
    assert orbit.shape == (jspam.timing(PARAMS)["n_steps"] + 1, 7)
    np.testing.assert_array_equal(orbit[-1, :6], final[-1])
    assert 0 <= orbit[-1, 6] < 0.0025  # 1 to 2 steps past now


def test_orbit_closest_approach_matches_timing():
    timing = jspam.timing(PARAMS)
    orbit = jspam.run_orbit(PARAMS)
    separation = np.linalg.norm(orbit[:, :3], axis=1)
    closest = separation.argmin()
    np.testing.assert_allclose(separation[closest], timing["closest_distance"], rtol=1e-3)
    np.testing.assert_allclose(orbit[closest, 6], timing["t_closest"], atol=0.01)


def test_orbit_uses_the_mass_model():
    heavy = jspam.run_orbit(PARAMS, mass_model={"r_scale": 5.0}, normalization="total")
    plain = jspam.run_orbit(PARAMS, n_steps=len(heavy) - 1, normalization="total")
    assert not np.allclose(heavy[-1, :3], plain[-1, :3])


# --- The search path and a given start time ----------------------------------------------------------


def test_search_orbit_goes_back_from_now():
    path = jspam.search_orbit(PARAMS)
    assert abs(len(path) - 30001) <= 2
    np.testing.assert_allclose(path[0, :6], PARAMS[:6])  # now, as given
    assert path[0, 6] == 0.0
    assert -30.002 < path[-1, 6] <= -30.0 + 0.001
    assert np.all(np.diff(path[:, 6]) < 0)


def test_search_orbit_closest_point_matches_timing():
    timing = jspam.timing(PARAMS)
    separation = np.linalg.norm(jspam.search_orbit(PARAMS)[:, :3], axis=1)
    np.testing.assert_allclose(separation.min(), timing["closest_distance"], rtol=1e-12)


def test_run_starts_on_the_search_path():
    timing = jspam.timing(PARAMS)
    path = jspam.search_orbit(PARAMS)
    # Placement steps back while t_start < t, so it takes ceil(-t_start / step) steps (one more than the
    # nearest row when t_start isn't a whole number of steps)
    row = int(np.ceil(-timing["t_start"] / jspam.STEP))
    start = jspam.run_orbit(PARAMS, n_steps=0)[0]
    np.testing.assert_allclose(start[:6], path[row, :6], rtol=1e-9, atol=1e-9)


def test_given_start_time():
    timing = jspam.timing(PARAMS, t_start=-8.0)
    assert timing["t_start"] == -8.0
    assert timing["n_steps"] == 8002
    assert not timing["start_fallback"]
    orbit = jspam.run_orbit(PARAMS, t_start=-8.0)
    assert orbit[0, 6] == -8.0
    np.testing.assert_allclose(orbit[-1, :6], PARAMS[:6], atol=0.02)  # still ends now
    path = jspam.search_orbit(PARAMS)
    np.testing.assert_allclose(orbit[0, :6], path[8000, :6], rtol=1e-9, atol=1e-9)


def test_start_time_does_not_leak_into_the_next_run():
    default = jspam.run_collision(PARAMS, n1=5, n2=5, seed=1)
    jspam.run_collision(PARAMS, n1=5, n2=5, seed=1, t_start=-8.0)
    np.testing.assert_array_equal(jspam.run_collision(PARAMS, n1=5, n2=5, seed=1), default)


def test_rejects_a_start_after_now():
    with pytest.raises(ValueError):
        jspam.timing(PARAMS, t_start=0.5)


def test_initial_state_at_now_is_unperturbed():
    start = jspam.initial_state(PARAMS, n1=50, n2=50, seed=3)
    now = jspam.initial_state(PARAMS, n1=50, n2=50, seed=3, t_start=0.0)
    # The secondary sits where the parameters put it, with no steps back
    np.testing.assert_array_equal(now[-1], PARAMS[:6])
    np.testing.assert_array_equal(now[:50], start[:50])  # the primary's disk is the same
    # The secondary's disk is the same draw, moved with its center
    np.testing.assert_allclose(now[50:100] - now[-1], start[50:100] - start[-1], rtol=1e-12, atol=1e-12)


def test_only_initial_state_accepts_a_start_of_now():
    with pytest.raises(ValueError):
        jspam.run_collision(PARAMS, n1=5, n2=5, seed=1, t_start=0.0)
    with pytest.raises(ValueError):
        jspam.initial_state(PARAMS, n1=5, n2=5, seed=1, t_start=0.5)
