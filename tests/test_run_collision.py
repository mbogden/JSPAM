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
