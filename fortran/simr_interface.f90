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

  ! Run one collision and return every particle at the end.
  ! Call set_seed or seed_from_system first: this doesn't seed the generator itself.
  subroutine run_collision(collision_param, potential, npts1, npts2, heat_1, heat_2, final_state)

    real (kind=8), dimension(22), intent(in) :: collision_param
    integer, intent(in) :: potential, npts1, npts2
    real (kind=8), intent(in) :: heat_1, heat_2
    real (kind=8), dimension(npts1+npts2+1,6), intent(out) :: final_state

    integer (kind=4) :: nstep_local, i

    call setup_run(collision_param, potential, npts1, npts2, heat_1, heat_2)

    nstep_local = int((tend - tstart) / h) + 2
    do i = 1, nstep_local
      call TAKE_A_STEP
    enddo

    final_state = x0
    call end_run

  end subroutine run_collision

  ! Set up one collision and return every particle at the start, before any step is taken.
  ! The same inputs and seed give exactly the starting state run_collision steps from.
  subroutine initial_state(collision_param, potential, npts1, npts2, heat_1, heat_2, start_state)

    real (kind=8), dimension(22), intent(in) :: collision_param
    integer, intent(in) :: potential, npts1, npts2
    real (kind=8), intent(in) :: heat_1, heat_2
    real (kind=8), dimension(npts1+npts2+1,6), intent(out) :: start_state

    call setup_run(collision_param, potential, npts1, npts2, heat_1, heat_2)
    start_state = x0
    call end_run

  end subroutine initial_state

  ! Shared by run_collision and initial_state: sets SPAM's module variables the way basic_run does for a
  ! model string, then places both galaxies' particles at the start time.
  subroutine setup_run(collision_param, potential, npts1, npts2, heat_1, heat_2)

    real (kind=8), dimension(22), intent(in) :: collision_param
    integer, intent(in) :: potential, npts1, npts2
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
