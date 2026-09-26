"""캔위성(CanSat) 낙하산 설계·하강 시뮬레이션 패키지.

- parachute_parameter : 설계 입력 (캔위성, 캐노피, 사출 조건, 발사장)
- parachute_model     : 설계 파라미터 -> Cd / CdS 모델 (Knacke 기반), 사이징, 개방하중
- simul               : RocketPy 기반 하강 시뮬레이션
- util                : 해석식, 바람 변환, 그래프
"""

from . import util
from .parachute_model import (
    CANOPY_TABLE,
    CdBreakdown,
    MassModel,
    canopy_mass,
    design_cd,
    opening_load_bound,
    packed_volume,
    predict_descent_rate,
    size_for_descent_rate,
)
from .parachute_parameter import (
    CANOPY_SHAPES, SCENARIOS, CanSat, Canopy, Deployment, Site, esa_cansat, korea_cansat,
)
from .simul import DescentResult, build_body, build_environment, initial_velocity, release_offset, run_descent

__all__ = [
    "CANOPY_SHAPES", "CANOPY_TABLE", "SCENARIOS", "CanSat", "Canopy", "Deployment", "Site",
    "korea_cansat", "esa_cansat", "initial_velocity", "release_offset",
    "CdBreakdown", "design_cd", "predict_descent_rate", "size_for_descent_rate", "opening_load_bound",
    "MassModel", "canopy_mass", "packed_volume",
    "DescentResult", "run_descent", "build_environment", "build_body", "util",
]
