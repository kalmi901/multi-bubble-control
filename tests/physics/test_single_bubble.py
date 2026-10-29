import numpy as np
from numpy.testing import assert_allclose

from multibubble.physics import MaterialProperties
from multibubble.physics.acoustic_field import AcousticField, UniformHarmonic
from multibubble.physics.common import ReferenceScales
from multibubble.physics.single_bubble import (
    BubbleParameters,
    prepare_single_bubble_parameters,
    single_bubble_rhs,
)


def test_unforced_equilibrium_has_zero_rhs():
    material = MaterialProperties()

    bubble = BubbleParameters(
        equilibrium_radius=50e-6,
    )

    field = AcousticField(
        components=(
            UniformHarmonic(
                pressure_amplitude=0.0,
                frequency=20e3,
            ),
        )
    )

    reference = ReferenceScales(
        radius=100e-6,
        frequency=20e3,
    )

    reference_radius = 100e-6

    params = prepare_single_bubble_parameters(material, bubble, field, reference, 1)

    y = np.array(
        [
            [
                bubble.equilibrium_radius / reference_radius,
                0.0,
            ]
        ],
        dtype=np.float64,
    )

    t = np.array([0.0])

    dydt = single_bubble_rhs(
        t,
        y,
        params,
    )

    assert dydt.shape == y.shape

    assert_allclose(
        dydt,
        np.zeros_like(y),
        rtol=0.0,
        atol=1e-13,
    )


def test_rhs_preserves_batch_shape():
    material = MaterialProperties()

    bubble = BubbleParameters(
        equilibrium_radius=np.array([40e-6, 50e-6, 60e-6]),
    )

    frequencies = np.array([20e3, 25e3, 30e3], dtype=np.float64)

    field = AcousticField(
        components=(
            UniformHarmonic(
                pressure_amplitude=0.0,
                frequency=frequencies,
            ),
        )
    )

    reference = ReferenceScales(
        radius=100e-6,
        frequency=frequencies,
    )

    params = prepare_single_bubble_parameters(material, bubble, field, reference, len(frequencies))

    y = np.array(
        [
            [0.4, 0.0],
            [0.5, 0.0],
            [0.6, 0.0],
        ],
        dtype=np.float64,
    )

    t = np.zeros(3, dtype=np.float64)

    dydt = single_bubble_rhs(t, y, params)

    assert dydt.shape == y.shape


def test_radius_derivative_equals_wall_velocity():
    material = MaterialProperties()

    bubble = BubbleParameters(
        equilibrium_radius=np.array([40e-6, 50e-6, 60e-6]),
    )

    frequencies = np.array(
        [20e3, 25e3, 30e3],
    )

    field = AcousticField(
        components=(
            UniformHarmonic(
                pressure_amplitude=0.0,
                frequency=frequencies,
            ),
        )
    )

    reference = ReferenceScales(
        radius=100e-6,
        frequency=frequencies,
    )

    params = prepare_single_bubble_parameters(material, bubble, field, reference, 3)

    y = np.array(
        [
            [0.8, 0.1],
            [1.1, -0.2],
            [0.9, 0.35],
        ],
        dtype=np.float64,
    )

    t = np.zeros(3, dtype=np.float64)

    dydt = single_bubble_rhs(t, y, params)

    np.testing.assert_allclose(
        dydt[:, 0],
        y[:, 1],
    )


def test_environment_specific_forcing_amplitude():
    material = MaterialProperties()

    bubble = BubbleParameters(
        equilibrium_radius=50e-6,
    )

    amplitude = np.array([0.0, 0.5e5, 1.0e5], dtype=np.float64)

    field = AcousticField(
        components=(
            UniformHarmonic(
                pressure_amplitude=amplitude,
                frequency=25e3,
            ),
        )
    )

    reference = ReferenceScales(
        radius=100e-6,
        frequency=25e3,
    )

    params = prepare_single_bubble_parameters(material, bubble, field, reference, len(amplitude))

    # All environments start from the same equilibrium state.
    y = np.array(
        [
            [0.5, 0.0],
            [0.5, 0.0],
            [0.5, 0.0],
        ],
        dtype=np.float64,
    )

    # Quarter acoustic cycle:
    # sin(2*pi*t) = 1
    # cos(2*pi*t) = 0
    #
    # This removes the pressure-derivative contribution and isolates
    # the forcing-amplitude term.
    t = np.full(3, 0.25, dtype=np.float64)

    dydt = single_bubble_rhs(t, y, params)

    # No radial motion initially.
    np.testing.assert_allclose(
        dydt[:, 0],
        0.0,
        atol=1e-14,
    )

    # Environment 0 has zero forcing and remains at equilibrium.
    np.testing.assert_allclose(
        dydt[0, 1],
        0.0,
        atol=1e-12,
    )

    # The two driven environments must respond differently.
    assert dydt[1, 1] != 0.0
    assert dydt[2, 1] != 0.0

    # At this state and phase the forcing contribution is linear
    # in pressure amplitude, so doubling p_A doubles acceleration.
    np.testing.assert_allclose(
        dydt[2, 1],
        2.0 * dydt[1, 1],
        rtol=1e-12,
        atol=1e-12,
    )


def test_rhs_satisfies_dimensional_keller_miksis_equation():
    """Verify the dimensionless RHS against the dimensional KM equation."""

    material = MaterialProperties(
        density=998.0,
        sound_speed=1500.0,
        surface_tension=0.0725,
        dynamic_viscosity=0.001,
        vapor_pressure=0.0,
        ambient_pressure=1.013e5,
        polytropic_exponent=1.4,
    )

    bubble = BubbleParameters(
        equilibrium_radius=50e-6,
    )

    field = AcousticField(
        components=(
            UniformHarmonic(
                pressure_amplitude=0.3e5, frequency=np.array([25e3, 25e3, 25e3]), temporal_phase=0.3
            ),
        )
    )

    # Deliberately choose R_ref != R0 so that the test also checks
    # the radius/velocity/acceleration scaling.
    reference = ReferenceScales(
        radius=80e-6,
        frequency=25e3,
    )

    params = prepare_single_bubble_parameters(
        material,
        bubble,
        field,
        reference,
        3,
    )

    # ---------------------------------------------------------------
    # Independent physical states
    # ---------------------------------------------------------------

    tau = np.array(
        [0.13, 0.37, 0.81],
        dtype=np.float64,
    )

    radius_ratio = np.array(
        [0.85, 1.05, 1.20],
        dtype=np.float64,
    )

    wall_velocity = np.array(
        [-2.0, 0.7, 1.5],
        dtype=np.float64,
    )

    R0 = bubble.equilibrium_radius
    frequency = field.components[0].frequency

    radius = R0 * radius_ratio

    # Dimensionless state used by single_bubble_rhs:
    #
    #   R* = R / R_ref
    #
    #   tau = f t
    #
    #   U* = dR*/dtau = Rdot / (R_ref f)
    #
    y = np.column_stack(
        (
            radius / reference.radius,
            wall_velocity / (reference.radius * frequency),
        )
    )

    dydt = single_bubble_rhs(
        tau,
        y,
        params,
    )

    # ---------------------------------------------------------------
    # Convert dimensionless acceleration back to SI units.
    #
    # U* = Rdot / (R_ref f)
    #
    # dU*/dtau = Rddot / (R_ref f^2)
    # ---------------------------------------------------------------

    wall_acceleration = dydt[:, 1] * reference.radius * frequency**2

    # ---------------------------------------------------------------
    # Independently evaluate the DIMENSIONAL Keller-Miksis equation.
    #
    # Nothing below uses the prepared dimensionless coefficients.
    # ---------------------------------------------------------------

    rho = material.density
    c = material.sound_speed
    sigma = material.surface_tension
    mu = material.dynamic_viscosity
    p_vapor = material.vapor_pressure
    p_ambient = material.ambient_pressure
    kappa = material.polytropic_exponent

    p_amplitude = field.components[0].pressure_amplitude
    omega = 2.0 * np.pi * frequency
    phase = field.components[0].temporal_phase

    theta = 2.0 * np.pi * tau + phase

    acoustic_pressure = p_amplitude * np.sin(theta)

    acoustic_pressure_dt = p_amplitude * omega * np.cos(theta)

    # Equilibrium gas pressure.
    gas_pressure_0 = p_ambient - p_vapor + 2.0 * sigma / R0

    gas_pressure = gas_pressure_0 * (R0 / radius) ** (3.0 * kappa)

    # Pressure difference acting on the wall:
    #
    # q = p_g - (p_0 - p_v)
    #     - 2 sigma / R
    #     - 4 mu Rdot / R
    #     - p_a
    #
    pressure_difference = (
        gas_pressure
        - (p_ambient - p_vapor)
        - 2.0 * sigma / radius
        - 4.0 * mu * wall_velocity / radius
        - acoustic_pressure
    )

    # Full time derivative dq/dt.
    #
    # Important: this contains Rddot through the viscous term.
    #
    pressure_difference_dt = (
        -3.0 * kappa * gas_pressure * wall_velocity / radius
        + 2.0 * sigma * wall_velocity / radius**2
        - 4.0 * mu * (wall_acceleration / radius - wall_velocity**2 / radius**2)
        - acoustic_pressure_dt
    )

    # ---------------------------------------------------------------
    # Original Keller-Miksis equation
    # ---------------------------------------------------------------

    lhs = (1.0 - wall_velocity / c) * radius * wall_acceleration + 1.5 * (
        1.0 - wall_velocity / (3.0 * c)
    ) * wall_velocity**2

    rhs = (
        1.0 + wall_velocity / c
    ) * pressure_difference / rho + radius * pressure_difference_dt / (rho * c)

    np.testing.assert_allclose(
        lhs,
        rhs,
        rtol=1e-11,
        atol=1e-9,
    )
