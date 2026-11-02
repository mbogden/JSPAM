project('jspam', 'c', 'fortran', version: '0.1.0')

py = import('python').find_installation(pure: false)

incdir_numpy = run_command(py, '-c', 'import numpy; print(numpy.get_include())', check: true).stdout().strip()
incdir_f2py = run_command(py, '-c', 'import numpy.f2py; print(numpy.f2py.get_include())', check: true).stdout().strip()

# f2py only reads simr_interface.f90: Python sees that file's subroutines, not the original modules.
f2py_output = custom_target(
  '_spam-f2py',
  input: 'fortran/simr_interface.f90',
  output: ['_spammodule.c', '_spam-f2pywrappers2.f90'],
  command: [py, '-m', 'numpy.f2py', '@INPUT@', '-m', '_spam', '--lower', '--build-dir', '@OUTDIR@'],
)

py.extension_module(
  '_spam',
  [
    'fortran/parameters_module.f90',
    'fortran/io_module.f90',
    'fortran/df_module.f90',
    'fortran/setup_module.f90',
    'fortran/integrator.f90',
    'fortran/init_module.f90',
    'fortran/simr_interface.f90',
    f2py_output,
    incdir_f2py / 'fortranobject.c',
  ],
  include_directories: include_directories(incdir_numpy, incdir_f2py),
  fortran_args: ['-std=legacy'],  # the original modules use older Fortran syntax
  install: true,
  subdir: 'jspam',
)

py.install_sources('src/jspam/__init__.py', subdir: 'jspam')
