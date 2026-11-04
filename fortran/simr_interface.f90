! Calls SPAM from Python. Added for the SIMR project; not part of the original JSPAM.
! Sets up a run the way basic_run does when it's given a model string, without reading the command
! line or writing files. Every call passes everything a run depends on, so no call can pick up values
! left over from an earlier one.
module simr_interface

  use parameters_module
  use init_module
  use integrator
  use df_module

  implicit none

contains

  ! The number of integers random_seed needs on this compiler.
  subroutine random_seed_size(size_out)
    integer, intent(out) :: size_out
    call random_seed(size=size_out)
  end subroutine random_seed_size

  ! When a run starts, and how many steps take it to its normal end, without integrating it.
  subroutine get_timing(collision_param, potential, lnl_in, r_scale, m_halo, m_bulge, h_bulge, h_disk, &
       normalization, t_start, n_steps, start_fallback)

    real (kind=8), dimension(22), intent(in) :: collision_param
    integer, intent(in) :: potential, normalization
    real (kind=8), intent(in) :: lnl_in, r_scale, m_halo, m_bulge, h_bulge, h_disk
    real (kind=8), intent(out) :: t_start
    integer, intent(out) :: n_steps
    integer, intent(out) :: start_fallback

    call setup_run(collision_param, potential, lnl_in, r_scale, m_halo, m_bulge, h_bulge, h_disk, &
         normalization, 0, 0, 0.0d0, 0.0d0, 0)
    t_start = tstart
    n_steps = int((tend - tstart) / h) + 2   ! Runs normally go 1 to 2 steps past t = 0, as in basic_run
    start_fallback = 0
    if (tstart_fallback) start_fallback = 1
    call cleanup_run

  end subroutine get_timing

  ! The secondary galaxy's center at every step, from the start through n_steps.
  ! Each row is x, y, z, vx, vy, vz, t. No particles, so no random numbers.
  subroutine run_orbit(collision_param, potential, lnl_in, r_scale, m_halo, m_bulge, h_bulge, h_disk, &
       normalization, n_steps, orbit)

    real (kind=8), dimension(22), intent(in) :: collision_param
    integer, intent(in) :: potential, normalization, n_steps
    real (kind=8), intent(in) :: lnl_in, r_scale, m_halo, m_bulge, h_bulge, h_disk
    real (kind=8), dimension(n_steps+1,7), intent(out) :: orbit

    integer :: step

    call setup_run(collision_param, potential, lnl_in, r_scale, m_halo, m_bulge, h_bulge, h_disk, &
         normalization, 0, 0, 0.0d0, 0.0d0, 0)

    orbit(1,1:6) = x0(n,1:6)
    orbit(1,7) = tstart
    do step = 1, n_steps
      call TAKE_A_STEP
      orbit(step+1,1:6) = x0(n,1:6)
      orbit(step+1,7) = tstart + step*hbase
    enddo

    call cleanup_run

  end subroutine run_orbit

  ! One simulation, saved at each step in end_steps (ascending; step 0 is the initial state).
  ! snapshots(:,:,k) is every particle at end_steps(k); the last row is the secondary's center.
  subroutine run_particles(collision_param, potential, lnl_in, r_scale, m_halo, m_bulge, h_bulge, h_disk, &
       normalization, npts1, npts2, heat_1, heat_2, heat_mode, seed_size, seed_values, n_ends, end_steps, &
       snapshots, times)

    real (kind=8), dimension(22), intent(in) :: collision_param
    integer, intent(in) :: potential, normalization, npts1, npts2, heat_mode, seed_size, n_ends
    real (kind=8), intent(in) :: lnl_in, r_scale, m_halo, m_bulge, h_bulge, h_disk, heat_1, heat_2
    integer, dimension(seed_size), intent(in) :: seed_values
    integer, dimension(n_ends), intent(in) :: end_steps
    real (kind=8), dimension(npts1+npts2+1,6,n_ends), intent(out) :: snapshots
    real (kind=8), dimension(n_ends), intent(out) :: times

    integer :: step, k

    call random_seed(put=seed_values)
    call setup_run(collision_param, potential, lnl_in, r_scale, m_halo, m_bulge, h_bulge, h_disk, &
         normalization, npts1, npts2, heat_1, heat_2, heat_mode)

    k = 1
    if (end_steps(1) == 0) then
      snapshots(:,:,1) = x0
      times(1) = tstart
      k = 2
    endif

    do step = 1, end_steps(n_ends)
      call TAKE_A_STEP
      if (k <= n_ends) then
        if (end_steps(k) == step) then
          snapshots(:,:,k) = x0
          times(k) = tstart + step*hbase
          k = k + 1
        endif
      endif
    enddo

    call cleanup_run

  end subroutine run_particles

  ! The mass model last built, to check that each value reached its own Fortran variable.
  subroutine get_built_mass_model(lnl_out, r_scale, m_halo, m_bulge, h_bulge, h_disk, normalization)
    real (kind=8), intent(out) :: lnl_out, r_scale, m_halo, m_bulge, h_bulge, h_disk
    integer, intent(out) :: normalization
    lnl_out = built_mass_model(1)
    r_scale = built_mass_model(2)
    m_halo = built_mass_model(3)
    m_bulge = built_mass_model(4)
    h_bulge = built_mass_model(5)
    h_disk = built_mass_model(6)
    normalization = built_normalization
  end subroutine get_built_mass_model

  subroutine setup_run(collision_param, potential, lnl_in, r_scale, m_halo, m_bulge, h_bulge, h_disk, &
       normalization, npts1, npts2, heat_1, heat_2, heat_mode)

    real (kind=8), dimension(22), intent(in) :: collision_param
    integer, intent(in) :: potential, normalization, npts1, npts2, heat_mode
    real (kind=8), intent(in) :: lnl_in, r_scale, m_halo, m_bulge, h_bulge, h_disk, heat_1, heat_2

    call STANDARD_GALAXY1(mass1, epsilon1, rin1, rout1, rscale1, theta1, phi1, opt1, heat1)
    call STANDARD_GALAXY2(mass2, epsilon2, rin2, rout2, rscale2, theta2, phi2, opt2, heat2)
    call TEST_COLLISION(n, n1, n2, time, inclination_degree, omega_degree, &
         rmin, velocity_factor, h, nstep, nout)

    ! Same order as parse_state_info_string
    sec_vec = collision_param(1:6)
    mass1 = collision_param(7)
    mass2 = collision_param(8)
    rout1 = collision_param(9)
    rout2 = collision_param(10)
    phi1 = collision_param(11)
    phi2 = collision_param(12)
    theta1 = collision_param(13)
    theta2 = collision_param(14)
    epsilon1 = collision_param(15)
    epsilon2 = collision_param(16)
    rscale1 = collision_param(17:19)
    rscale2 = collision_param(20:22)
    use_sec_vec = .true.

    heat1 = heat_1
    heat2 = heat_2
    heat_offset = 0.0d0
    if (heat_mode == 1) heat_offset = 0.5d0

    ! Same values as custom_collision uses for a model string
    potential_type = potential
    h = hbase
    tstart = -5
    tend = 0
    time = -5
    tIsSet = .false.

    n1 = npts1
    n2 = npts2
    n = n1 + n2
    eps1 = epsilon1*epsilon1
    eps2 = epsilon2*epsilon2

    allocate(x0(n+1,6))
    allocate(xout(n+1,6))
    allocate(projected(n+1,3))

    call init_mass_model_if_changed(lnl_in, r_scale, m_halo, m_bulge, h_bulge, h_disk, normalization)
    call place_galaxies
    call INIT_RKVAR(x0, mass1, mass2, epsilon1, epsilon2, theta1, phi1, &
         theta2, phi2, rscale1, rscale2, rout1, rout2, n, n1, n2)

  end subroutine setup_run

  subroutine cleanup_run
    ! DEALLOCATE_RKVAR doesn't free these, which would stop a second run in the same process
    deallocate(ival11, ival22, ivaln, df_force11, df_force22, df_forcen, c3n)
    call DEALLOCATE_RKVAR
    deallocate(x0)
    deallocate(xout)
    deallocate(projected)
  end subroutine cleanup_run

end module simr_interface
