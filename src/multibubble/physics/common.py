from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
Parameter = float | FloatArray


@dataclass(frozen=True)
class MaterialProperties:
    density: Parameter = 998.0
    sound_speed: Parameter = 1500.0
    surface_tension: Parameter = 0.0725
    dynamic_viscosity: Parameter = 0.001
    vapor_pressure: Parameter = 0.0
    ambient_pressure: Parameter = 1.013e5
    polytropic_exponent: Parameter = 1.4


@dataclass(frozen=True)
class AcousticForcing:
    pressure_amplitude: Parameter
    frequency: Parameter
    phase: Parameter = 0.0
