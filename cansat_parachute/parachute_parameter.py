"""설계 입력 파라미터 (SI 단위) 와 기준 시나리오.

기본값은 국내 '캔위성 체험·경연대회' (우주항공청·KAIST 인공위성연구소) 창작부 기준으로 고정했다.
  - 규정(cansat.kaist.ac.kr 대회규정): 발사관 내부 Ø110 × 200 mm, 캔위성 용기 Ø83 × 115 mm,
    총질량 창작부 ≤ 1000 g / 슬기부 ≤ 800 g (낙하산 포함), 예정 발사고도 창작부 350 m / 슬기부 250 m,
    낙하산은 로켓 분리 즉시 전개. 하강속도·시간 규정 없음.
  - 장소/시기: 전남 고흥 항공센터 (34.611 N, 127.206 E, 해발 ≈ 1 m 간척지), 8월 초.
  - 기후(기상청 고흥 1991~2020 8월 평년): 일최고 30.1 °C, 평균 25.7 °C, 상대습도 77 %.
    -> 오전~정오 발사 가정 29 °C, 1008 hPa, RH 77 % (공기밀도 ≈ 1.15 kg/m^3, 표준대기보다 6 % 희박)
  - 바람: 평년 풍속 통계를 확보하지 못해 10 m 풍속 3 m/s, 남남서풍(200°)을 설계 기준으로 가정.
    강풍 케이스는 gust_factor = 2 (6 m/s). 고도 프로파일은 멱법칙 지수 0.14 (개활지).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

CANOPY_SHAPES = ("flat_circular", "conical", "hemispherical", "extended_skirt", "cross")


@dataclass
class CanSat:
    """캔위성 본체. 전개 전 자유낙하와 후류 손실 계산에 쓰인다."""

    mass: float = 0.90         # kg, 낙하산 포함 총질량. 규정 상한: 창작부 1.0 kg, 슬기부 0.8 kg
    diameter: float = 0.083    # m, 대회 캔위성 용기 Ø83 mm
    height: float = 0.115      # m, 용기 높이 115 mm
    cd_body: float = 1.0       # 캔 자체 항력계수 (전개 전 자유낙하 구간)
    stopping_distance: float = 0.03  # m, 착지 완충 거리 (캔 변형 + 완충재 + 지면 침하).
    #                                  콘크리트에 단단한 캔 0.5~1 cm, 잔디/흙 2~5 cm, 폼 완충재 3~8 cm

    @property
    def frontal_area(self) -> float:
        return math.pi * self.diameter**2 / 4


@dataclass
class Canopy:
    """낙하산 설계 파라미터. 모든 Cd 는 공칭면적 S0 = pi*D0^2/4 기준이다.

    D0 는 벤트를 포함한 천 전체 면적으로 정의한 공칭 지름이다.
    반구형은 표면적 2*pi*R^2 기준이므로 D0 = 2.83 R 이다 (투영지름 2R 이 아님).
    """

    shape: str = "cross"
    diameter: float = 0.75               # m, 공칭 지름 D0 (0.9 kg, 십자형, 약 7 m/s 하강 기준)
    line_ratio: float = 1.0              # Le/D0, 현수선 길이 / 공칭 지름 (권장 1.0~1.5)
    vent_ratio: float = 0.0025           # 벤트 면적 / S0 (Knacke 권장 0.25 %, 1 % 는 과다)
    riser_length: float = 0.30           # m, 캔 ~ 현수선 합류점
    rigging_length: float | None = None  # m, 캔 ~ 캐노피 스커트 거리. None 이면 riser + Le
    cd_override: float | None = None     # 낙하시험으로 측정한 Cd (S0 기준). 주면 모델 대신 사용
    cd_level: str = "mid"                # 표 범위 중 어느 값을 쓸지: "low" | "mid" | "high"

    def __post_init__(self):
        if self.shape not in CANOPY_SHAPES:
            raise ValueError(f"shape 는 {CANOPY_SHAPES} 중 하나여야 합니다: {self.shape!r}")
        if self.cd_level not in ("low", "mid", "high"):
            raise ValueError(f"cd_level 은 low/mid/high 중 하나여야 합니다: {self.cd_level!r}")

    @property
    def area(self) -> float:
        """공칭면적 S0 [m^2]"""
        return math.pi * self.diameter**2 / 4

    @property
    def line_length(self) -> float:
        """현수선 길이 Le [m]"""
        return self.line_ratio * self.diameter

    @property
    def effective_rigging_length(self) -> float:
        """캔에서 캐노피 스커트까지 거리 [m]"""
        if self.rigging_length is not None:
            return self.rigging_length
        return self.riser_length + self.line_length


@dataclass
class Deployment:
    """사출/전개 조건. 로켓 정점에서 분리되어 즉시 낙하산이 펴지는 것이 규정이다."""

    altitude_agl: float = 350.0            # m, 분리 고도 = 로켓 정점. 창작부 350, 슬기부 250, 공고문 300~500
    horizontal_speed: float = 5.0          # m/s, 분리 시 수평 속력. 로켓 웨더콕(맞바람 쪽 기울기) + 사출 속도
    horizontal_heading_deg: float | None = None  # 진행 방향 (북 0°, 시계). None 이면 바람 불어오는 쪽(맞바람)
    vz0: float = 0.0                       # m/s, 정점이므로 수직 속도 0
    lag: float = 1.0                       # s, 분리 후 완전 개방까지 (라인 스트레치 + 팽창)
    trigger_altitude_agl: float | None = None  # None: 분리 즉시 전개 (규정). 값: 그 고도까지 자유낙하 후 전개
    apogee_offset_m: float = 0.0           # m, 분리 지점의 발사 지점 대비 수평 거리 (맞바람 쪽). 로켓 웨더콕 시 50~150 m


@dataclass
class Site:
    """발사장과 대기 조건. 지표 관측값으로 온도·압력 프로파일을, 10 m 풍속으로 바람 프로파일을 만든다."""

    name: str = "고흥 항공센터"
    latitude: float = 34.60795        # deg, 발사 지점 (비행장 경계 폴리곤 도심, 가정)
    longitude: float = 127.21046      # deg
    elevation: float = 1.0            # m, 해안 간척지
    geometry_file: str | None = "data/goheung_site.geojson"  # 부지 경계·활주로·해안선 (OpenStreetMap). None 이면 지도 생략
    surface_temperature_c: float = 29.0   # °C, 8월 오전~정오 (평년 일최고 30.1, 평균 25.7)
    surface_pressure_hpa: float = 1008.0  # hPa, 여름철 전형값
    relative_humidity: float = 0.77       # 8월 평년 77 %
    wind_speed_10m: float = 3.0           # m/s, 10 m 기준풍속 (가정값, 기상청 통계로 교체 권장)
    wind_from_deg: float = 200.0          # 불어오는 방향. 여름 남남서풍
    wind_shear_exponent: float = 0.14     # 멱법칙 지수 (개활지)
    gust_factor: float = 1.0              # 프로파일 전체 배율. 강풍 설계 케이스 2.0

    @property
    def wind_speed(self) -> float:
        return self.wind_speed_10m


# ------------------------------------------------------------------ 기준 시나리오
def korea_cansat(division: str = "창작부") -> tuple[CanSat, Deployment, Site]:
    """국내 캔위성 경연대회 기준 시나리오. division: '창작부'(대학, 350 m, ≤1.0 kg) | '슬기부'(고교, 250 m, ≤0.8 kg)"""
    cansat, deploy, site = CanSat(), Deployment(), Site()
    if division == "슬기부":
        cansat = replace(cansat, mass=0.75)
        deploy = replace(deploy, altitude_agl=250.0)
    elif division != "창작부":
        raise ValueError("division 은 '창작부' 또는 '슬기부'")
    return cansat, deploy, site


def esa_cansat() -> tuple[CanSat, Deployment, Site]:
    """참고용 ESA CanSat 조건: 66 × 115 mm, 300~350 g, 로켓 정점 약 1 km, 권장 하강속도 8~11 m/s. 대기는 표준대기."""
    cansat = CanSat(mass=0.35, diameter=0.066, height=0.115)
    deploy = Deployment(altitude_agl=1000.0)
    site = Site(name="ESA 기준 (표준대기)", elevation=0.0, surface_temperature_c=15.0,
                surface_pressure_hpa=1013.25, relative_humidity=0.0, geometry_file=None)
    return cansat, deploy, site


SCENARIOS = {
    "korea": lambda: korea_cansat("창작부"),
    "korea_high": lambda: korea_cansat("슬기부"),
    "esa": esa_cansat,
}
