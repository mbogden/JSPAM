! Calls SPAM from Python. Added for the SIMR project; not part of the original JSPAM.
! Uses the original modules unchanged and sets up a run the same way basic_run does when it's given
! a model string, without reading the command line or writing the particle output file.
module simr_interface

  use parameters_module
  use init_module
  use integrator

  implicit none

contains

  ! How many integers the compiler's random number generator needs for a seed.
  subroutine seed_size(nseed)
    integer, intent(out) :: nseed
    call RANDOM_SEED(size=nseed)
  end subroutine seed_size

  ! Set the generator's seed, so the disk particles start the same way every run.
  subroutine set_seed(values, nseed)
    integer, intent(in) :: nseed
    integer, dimension(nseed), intent(in) :: values
    call RANDOM_SEED(put=values)
  end subroutine set_seed

  ! Seed the generator from the operating system, so every run differs (the original behavior).
  subroutine seed_from_system()
    call RANDOM_SEED()
  end subroutine seed_from_system

  ! Set the mass model used by every run after this, until it's set again (Wallin's until then).
  ! values, in order: lnl, r_scale (rchalo), m_halo, m_bulge, h_bulge, h_disk.
  ! normalization 0: divide by Wallin's fixed 7.13333. 1: divide by this model's actual total mass.
  subroutine set_mass_model(values, normalization)
    real (kind=8), dimension(6), intent(in) :: values
    integer, intent(in) :: normalization
    mm_lnl = values(1)
    mm_rchalo = values(2)
    mm_mhalo = values(3)
    mm_mbulge = values(4)
    mm_hbulge = values(5)
    mm_hdisk = values(6)
    mass_normalization = normalization
  end subroutine set_mass_model

  ! Rows in the unit galaxy's tables (df_module's nnn).
  subroutine mass_profile_size(nrows)
    integer, intent(out) :: nrows
    nrows = nnn
  end subroutine mass_profile_size

  ! Build the unit galaxy from the current mass model and return its tables, row by row, as the
  ! integrator reads them: radius is the distance (SPAM units) at which df_index picks each row, and
  ! force is what a particle feels there. Masses are enclosed, for a galaxy of mass 1 in the 22
  ! parameters. With r_scale other than 10, the rows no longer match the table's own radii (Wallin's).
  subroutine mass_profile(nrows, radius, m_halo, m_disk, m_bulge, m_total, force)
    integer, intent(in) :: nrows
    real (kind=8), dimension(nrows), intent(out) :: radius, m_halo, m_disk, m_bulge, m_total, force
    integer :: j
    call INIT_DISTRIBUTION
    ! df_index: row = int(r * rs_internal / 100 * nnn) + 1, so row j starts at this distance
    do j = 1, nrows
      radius(j) = (j - 1) * 100.0d0 / (rs_internal * nnn)
    end do
    m_halo = mass_halo(1:nrows)
    m_disk = mass_disk(1:nrows)
    m_bulge = mass_bulge(1:nrows)
    m_total = mass_total(1:nrows)
    force = acceleration_particle(1:nrows) * rs_internal * rs_internal
  end subroutine mass_profile

  ! Run one collision and return every particle at the end.
  ! Call set_seed or seed_from_system first: this doesn't seed the generator itself.
  subroutine run_collision(collision_param, potential, npts1, npts2, heat_1, heat_2, heat_mode, final_state)

    real (kind=8), dimension(22), intent(in) :: collision_param
    integer, intent(in) :: potential, npts1, npts2, heat_mode
    real (kind=8), intent(in) :: heat_1, heat_2
    real (kind=8), dimension(npts1+npts2+1,6), intent(out) :: final_state

    integer (kind=4) :: nstep_local, i

    call setup_run(collision_param, potential, npts1, npts2, heat_1, heat_2, heat_mode)

    nstep_local = int((tend - tstart) / h) + 2
    do i = 1, nstep_local
      call TAKE_A_STEP
    enddo

    final_state = x0
    call end_run

  end subroutine run_collision

  ! Set up one collision and return every particle at the start, before any step is taken.
  ! The same inputs and seed give exactly the starting state run_collision steps from.
  subroutine initial_state(collision_param, potential, npts1, npts2, heat_1, heat_2, heat_mode, start_state)

    real (kind=8), dimension(22), intent(in) :: collision_param
    integer, intent(in) :: potential, npts1, npts2, heat_mode
    real (kind=8), intent(in) :: heat_1, heat_2
    real (kind=8), dimension(npts1+npts2+1,6), intent(out) :: start_state

    call setup_run(collision_param, potential, npts1, npts2, heat_1, heat_2, heat_mode)
    start_state = x0
    call end_run

  end subroutine initial_state

  ! Start every run after this at t_start (use = 1), or let SPAM's search choose (use = 0, the original).
  subroutine set_start_time(t_start, use)
    real (kind=8), intent(in) :: t_start
    integer, intent(in) :: use
    start_override = t_start
    use_start_override = (use == 1)
  end subroutine set_start_time

  ! The path SPAM's start-time search integrates: the secondary's center from now (t = 0) back to t = -30,
  ! one row per step (x, y, z, vx, vy, vz, t). Rows past nrows are unused. Ignores set_start_time.
  subroutine search_orbit(collision_param, potential, nmax, path, nrows)

    real (kind=8), dimension(22), intent(in) :: collision_param
    integer, intent(in) :: potential, nmax
    real (kind=8), dimension(nmax,7), intent(out) :: path
    integer, intent(out) :: nrows

    logical :: saved_use

    saved_use = use_start_override
    use_start_override = .false.
    call setup_run(collision_param, potential, 0, 0, 0.0d0, 0.0d0, 0)
    use_start_override = saved_use
    nrows = min(search_path_rows, nmax)
    path = 0.0d0
    path(1:nrows,:) = search_path(1:nrows,:)
    call end_run

  end subroutine search_orbit

  ! When a run starts and how many steps it takes, plus what the start-time search found. Only the two
  ! centers are set up (no particles), and nothing is stepped.
  ! search: getTStart's start time (before the -12 and near-0 overrides), the closest approach's distance,
  ! speed and time. start_fallback is 1 when an override replaced the start with -5.
  subroutine get_timing(collision_param, potential, t_start, n_steps, start_fallback, search)

    real (kind=8), dimension(22), intent(in) :: collision_param
    integer, intent(in) :: potential
    real (kind=8), intent(out) :: t_start
    integer, intent(out) :: n_steps, start_fallback
    real (kind=8), dimension(4), intent(out) :: search

    call setup_run(collision_param, potential, 0, 0, 0.0d0, 0.0d0, 0)
    t_start = tstart
    n_steps = int((tend - tstart) / h) + 2   ! the same count run_collision steps
    start_fallback = 0
    if (tstart_fallback) start_fallback = 1
    search = tstart_search
    call end_run

  end subroutine get_timing

  ! The secondary galaxy's center at the start and after each of n_steps steps.
  ! Each row is x, y, z, vx, vy, vz, t. No particles, so no random numbers are used.
  subroutine run_orbit(collision_param, potential, n_steps, orbit)

    real (kind=8), dimension(22), intent(in) :: collision_param
    integer, intent(in) :: potential, n_steps
    real (kind=8), dimension(n_steps+1,7), intent(out) :: orbit

    integer :: step

    call setup_run(collision_param, potential, 0, 0, 0.0d0, 0.0d0, 0)
    ! The secondary's center is x0's last row, x0(n,:): placing it (perturber_position_vec) did n = n + 1
    orbit(1,1:6) = x0(n,1:6)
    orbit(1,7) = time
    do step = 1, n_steps
      call TAKE_A_STEP
      orbit(step+1,1:6) = x0(n,1:6)
      orbit(step+1,7) = time
    enddo
    call end_run

  end subroutine run_orbit

  ! Shared by run_collision and initial_state: sets SPAM's module variables the way basic_run does for a
  ! model string, then places both galaxies' particles at the start time.
  subroutine setup_run(collision_param, potential, npts1, npts2, heat_1, heat_2, heat_mode)

    real (kind=8), dimension(22), intent(in) :: collision_param
    integer, intent(in) :: potential, npts1, npts2, heat_mode
    real (kind=8), intent(in) :: heat_1, heat_2

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

    ! Random motion added to each disk's particles. STANDARD_GALAXY1/2 set both to 0.
    heat1 = heat_1
    heat2 = heat_2
    ! heat_mode 0: Wallin's kicks, in [0, heat). 1: centered, in [-heat/2, heat/2), so no drift.
    if (heat_mode == 1) then
      heat_offset = 0.5d0
    else
      heat_offset = 0.0d0
    end if

    ! Same values as custom_collision uses for a model string
    potential_type = potential
    h = hbase
    tstart = -5
    tend = 0
    time = -5
    tIsSet = .false.
    tstart_search = 0.0d0
    tstart_fallback = .false.
    if (use_start_override) then   ! start at a given time instead of searching for one
      tstart = start_override
      time = start_override
      tIsSet = .true.
    end if

    n1 = npts1
    n2 = npts2
    n = n1 + n2
    eps1 = epsilon1*epsilon1
    eps2 = epsilon2*epsilon2

    allocate(x0(n+1,6))
    allocate(xout(n+1,6))
    allocate(projected(n+1,3))

    call CREATE_COLLISION
    call INIT_RKVAR(x0, mass1, mass2, epsilon1, epsilon2, theta1, phi1, &
         theta2, phi2, rscale1, rscale2, rout1, rout2, n, n1, n2)

  end subroutine setup_run

  ! Free what setup_run allocated, so the next run in the same process can start.
  subroutine end_run()

    ! DEALLOCATE_RKVAR doesn't free these, which would stop a second run in the same process
    deallocate(ival11, ival22, ivaln, df_force11, df_force22, df_forcen, c3n)
    call DEALLOCATE_RKVAR
    deallocate(x0)
    deallocate(xout)
    deallocate(projected)

  end subroutine end_run

end module simr_interface
