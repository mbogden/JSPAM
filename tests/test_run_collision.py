"""Checks the Python interface: the original program's answer, seeds, parameters and many endings."""

import numpy as np
import pytest

import jspam
from jspam import MassModel

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


# A made-up close encounter, inside the region where dynamical friction acts.
CLOSE_PARAMS = np.array(
    [
        1.5, 1.0, 0.5, -0.5, 0.5, 0.3, 1.0, 1.0, 1.0, 1.0, 30.0,
        60.0, 20.0, 40.0, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]
)  # fmt: skip


def final(params=PARAMS, **kwargs):
    snapshots, _, _ = jspam.run_particles(params, **{"seed": 1, "n1": 10, "n2": 5, **kwargs})
    return snapshots[-1]


def test_matches_original_program():
    snapshots, steps, times = jspam.run_particles(PARAMS, seed=1)
    assert snapshots.shape == (1, 2001, 6)
    assert np.isfinite(snapshots).all()
    np.testing.assert_allclose(snapshots[0, -1], EXPECTED_LAST_ROW, rtol=1e-5, atol=1e-6)
    assert steps[0] == jspam.get_timing(PARAMS).n_steps
    assert 0 < times[0] <= 0.002 + 1e-9


def test_same_seed_is_exactly_the_same():
    a = final(n1=50, n2=50, seed=7, heat1=0.1, heat2=0.1)
    b = final(n1=50, n2=50, seed=7, heat1=0.1, heat2=0.1)
    np.testing.assert_array_equal(a, b)


def test_different_seeds_differ_but_share_an_orbit():
    a = final(seed=1)
    b = final(seed=2)
    assert not np.array_equal(a[:-1], b[:-1])
    np.testing.assert_array_equal(a[-1], b[-1])


def test_a_run_between_two_others_changes_nothing():
    model = MassModel(lnl=0.01, m_halo=3.0, m_bulge=0.5, h_bulge=0.4, h_disk=1.5, normalization="total")
    a = final(seed=3)
    final(seed=4, n1=30, n2=2, mass_model=model, heat1=0.5, heat_mode="centered")
    np.testing.assert_array_equal(a, final(seed=3))


def test_particle_count_does_not_change_orbit():
    np.testing.assert_array_equal(final(n1=10, n2=5)[-1], final(n1=0, n2=0)[-1])


def test_orbit_matches_the_particle_run():
    orbit = jspam.run_orbit(PARAMS)
    timing = jspam.get_timing(PARAMS)
    assert orbit.shape == (timing.n_steps + 1, 7)
    assert orbit[0, 6] == timing.t_start
    np.testing.assert_array_equal(orbit[-1, :6], final()[-1])


@pytest.mark.parametrize(
    "value", [{"lnl": 0.02}, {"r_scale": 9.0}, {"m_halo": 5.0}, {"m_bulge": 0.5}, {"h_bulge": 0.3}, {"h_disk": 1.2}]
)
def test_every_mass_value_changes_the_orbit(value):
    # The old custom_runs.f90 set m_bulge, h_bulge and h_disk, then overwrote them with fixed values.
    # Particles and a close encounter, because lnl and h_bulge only matter close to a galaxy's center.
    changed = final(CLOSE_PARAMS, n1=50, n2=50, mass_model=MassModel(**value))
    assert not np.array_equal(changed, final(CLOSE_PARAMS, n1=50, n2=50))


def test_each_mass_value_reaches_its_own_variable():
    model = MassModel(lnl=0.011, r_scale=10.1, m_halo=5.1, m_bulge=0.31, h_bulge=0.21, h_disk=1.1, normalization="total")
    jspam.run_orbit(PARAMS, model)
    assert jspam.get_built_mass_model() == {
        "lnl": 0.011, "r_scale": 10.1, "m_halo": 5.1, "m_bulge": 0.31, "h_bulge": 0.21, "h_disk": 1.1, "normalization": 1,
    }  # fmt: skip


def test_normalization_changes_results_only_when_the_masses_change():
    wallin = jspam.run_orbit(PARAMS, MassModel(m_halo=2.9))
    total = jspam.run_orbit(PARAMS, MassModel(m_halo=2.9, normalization="total"))
    assert not np.array_equal(wallin[-1], total[-1]) or len(wallin) != len(total)
    # At the default masses the actual total is close to Wallin's 7.13333.
    default = jspam.run_orbit(PARAMS)
    default_total = jspam.run_orbit(PARAMS, MassModel(normalization="total"))
    if len(default) == len(default_total):
        np.testing.assert_allclose(default[-1, :3], default_total[-1, :3], rtol=0.05)


def test_heat_modes():
    args = {"n1": 2000, "n2": 0, "seed": 5, "end_steps": [0]}
    start = jspam.run_particles(PARAMS, **args)[0][0]
    wallin = jspam.run_particles(PARAMS, heat1=0.2, **args)[0][0]
    centered = jspam.run_particles(PARAMS, heat1=0.2, heat_mode="centered", **args)[0][0]
    # Same seed, same starting positions. Only the velocities differ.
    np.testing.assert_array_equal(start[:, :3], wallin[:, :3])
    np.testing.assert_array_equal(start[:, :3], centered[:, :3])
    wallin_kick = (wallin - start)[:-1, 3:]
    centered_kick = (centered - start)[:-1, 3:]
    np.testing.assert_allclose(wallin_kick.mean(axis=0), 0.1, atol=0.01)
    np.testing.assert_allclose(centered_kick.mean(axis=0), 0.0, atol=0.01)
    np.testing.assert_allclose(wallin_kick.std(axis=0), centered_kick.std(axis=0), rtol=1e-9)


def test_many_endings_match_separate_runs():
    timing = jspam.get_timing(PARAMS)
    end_steps = [0, timing.n_steps // 2, timing.n_steps, timing.n_steps + 500]
    snapshots, steps, times = jspam.run_particles(PARAMS, seed=9, n1=20, n2=10, end_steps=end_steps)
    assert snapshots.shape == (4, 31, 6)
    np.testing.assert_array_equal(steps, end_steps)
    np.testing.assert_allclose(times, timing.t_start + np.array(end_steps) * 0.001)
    for k, step in enumerate(end_steps):
        alone, _, _ = jspam.run_particles(PARAMS, seed=9, n1=20, n2=10, end_steps=[step])
        np.testing.assert_array_equal(snapshots[k], alone[0])


@pytest.mark.parametrize("end_steps", [[5, 5], [5, 3], [-1], [], [1.5]])
def test_rejects_bad_end_steps(end_steps):
    with pytest.raises(ValueError):
        jspam.run_particles(PARAMS, seed=1, end_steps=end_steps)


@pytest.mark.parametrize("seed", [-1, 1.5, None, True])
def test_rejects_bad_seeds(seed):
    with pytest.raises(ValueError):
        jspam.run_particles(PARAMS, seed=seed)


def test_seed_is_required():
    with pytest.raises(TypeError):
        jspam.run_particles(PARAMS)


def test_rejects_bad_parameters():
    with pytest.raises(ValueError):
        jspam.run_orbit(PARAMS[:5])
    with pytest.raises(ValueError):
        jspam.run_orbit(np.where(np.arange(22) == 0, np.nan, PARAMS))
    zero_velocity = PARAMS.copy()
    zero_velocity[3:6] = 0
    with pytest.raises(ValueError):
        jspam.run_orbit(zero_velocity)
    with pytest.raises(ValueError):
        MassModel(normalization="other").fortran_args()
    with pytest.raises(ValueError):
        MassModel(m_halo=0).fortran_args()
