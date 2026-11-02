# JSPAM: SIMR fork

> **This is a modified fork of [jfwallin/JSPAM](https://github.com/jfwallin/JSPAM) by John Wallin and
> collaborators.** It is still licensed under the Academic Free License version 3.0. Please cite the
> original JSPAM, as described in the original README below.

This fork makes SPAM's Fortran version installable and callable from Python, for use by
[galactic_tidal_constraints](https://github.com/mbogden/galactic_tidal_constraints). Maintained by
Matthew Ogden.

## Changes from the original
Compare with the original code using the `upstream-d6a61db` tag:
`git diff upstream-d6a61db simr-branch`

- Added Python packaging: `pyproject.toml`, `meson.build` and `src/jspam/`.
- Added `fortran/simr_interface.f90`, which runs the original modules from Python.
- Removed two debug `write` statements from `fortran/setup_module.f90`. They wrote about 80 MB of
  files per run and don't affect results.

Only the Fortran is packaged. The `java/`, `js/`, `python/` and `archive/` folders are unchanged.

## Install
You need Python 3.12 or newer, `gfortran` and a C compiler. Linux and WSL are supported; native
Windows is not.
```bash
pip install git+https://github.com/mbogden/JSPAM.git@simr-branch
```

## Use
```python
import numpy as np
import jspam

# The first 22 values of a model in archive/data.txt
model = (
    "-9.93853,-4.5805,15.43348,-1.54025,-2.97899,4.69234,34.41671,29.49607,2.97913,4.2501,"
    "301.0,35.5,310.80745,321.98757,0.3,0.3,0.0,0.0,0.0,0.0,0.0,0.0"
)
params = np.array(model.split(","), dtype=float)

final = jspam.run_collision(params, n1=1000, n2=1000)
print(final.shape)  # (2001, 6): x, y, z, vx, vy, vz for each particle
print(final[-1])  # the last row is the secondary galaxy's center
```
Disk particles start from unseeded random positions, so they differ between runs. The secondary
galaxy's orbit doesn't.

## Tests
```bash
pip install pytest
pytest tests
```

---

# Original README

JSPAM
=====

Software for simulating interacting galaxies using a restricted three-body approximation.

The current version of the code is available in the various language sub directories.  A portion of the historic version of the code is available in the archive directory.  The archive directory includes instructions for running a script that will recreate all 62 best-fit models from the Galaxy Zoo: Mergers project.  The js directory contains a web page that can run the same models in your browser.

The current language implementations include:

1. Fortran
2. Java
3. JavaScript
4. Python

Citations
---------
Please cite the Astrophysics Source Code Library (ASCL) entry for JSPAM, [http://ascl.net/1511.002](http://ascl.net/1511.002). The ASCL is also indexed by the Astrophysics Data System (ADS), see the entry for [JSPAM](http://adsabs.harvard.edu/abs/2015ascl.soft11002W).

The pre-print of our code release paper is available on [arxiv](http://arxiv.org/abs/1511.05041).  We will update the citation guidance once it has been fully accepted.

LICENSE
-------
The JSPAM software is free software which can be redistributed and/or modified under the terms of the [Academic Free License (”AFL”) v. 3.0.](https://opensource.org/licenses/AFL-3.0)
