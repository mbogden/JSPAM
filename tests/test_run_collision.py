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


def test_rejects_wrong_number_of_parameters():
    with pytest.raises(ValueError):
        jspam.run_collision(PARAMS[:5])
