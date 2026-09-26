"""설계 파라미터 -> Cd 모델.

출처: Knacke, T. W., *Parachute Recovery Systems Design Manual*, NWC TP 6575 (1992).
  - Table 5-1  solid textile 캐노피별 Cd0 범위, 개방하중계수 Cx, 진동각
  - Fig 5-18   불안정 캐노피의 하강속도별 Cd0 (수치는 Apogee Peak of Flight #668 이 그림에서 읽은 값)
  - Fig 5-19/20 현수선비 Le/D0 효과: 1.0->1.5 에서 +7 %, 1.5->2.0 에서 +3 %
  - Fig 5-21   전방동체 후류 손실: 거리 5 D 에서 -15 %, 10 D 에서 -8 %, 20 D 이상 무시
  - 5.2.4      레이놀즈수는 Cd 에 영향 없음 (박리 유동). 캔위성 Re ~ 2e5 도 해당
  - 6장        벤트 면적 0.25 % S0 권장, 1 % 는 과다

    Cd = Cd0(형상) * k_line(Le/D0) * k_vent(Av/S0) * k_wake(리깅 거리) [* k_vel(하강속도)]

모든 Cd 는 공칭면적 S0 = pi*D0^2/4 기준이다.
낙하시험 측정값(Canopy.cd_override)이 있으면 그 값을 그대로 쓴다. 측정값에는 모든 효과가 이미 들어 있다.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np

from .parachute_parameter import CANOPY_SHAPES, CanSat, Canopy
from .util import G0, required_cd_s, terminal_velocity


@dataclass(frozen=True)
class CanopyData:
    label: str
    cd0_range: tuple[float, float]        # Knacke Table 5-1, S0 기준
    cx: float                             # 무한질량 개방하중계수 (Table 5-1)
    oscillation_deg: tuple[float, float]  # 평균 진동각 범위
    dp_over_d0: float                     # 팽창 투영지름 / D0
    skirt_restricted: bool                # True: 현수선비 1.1 이상에서 Cd 이득 거의 없음 (Fig 5-20)
    unstable: bool                        # True: 저속에서 활공/진동으로 Cd 상승 (Fig 5-18)
    note: str


CANOPY_TABLE: dict[str, CanopyData] = {
    "flat_circular": CanopyData(
        "평면 원형", (0.75, 0.80), 1.7, (10, 40), 0.685, False, True,
        "제작 가장 쉬움. 진동 큼. 느릴수록 활공하며 Cd 상승",
    ),
    "conical": CanopyData(
        "원뿔형 25~30°", (0.75, 0.90), 1.8, (10, 30), 0.70, False, True,
        "같은 천으로 평면 대비 약 +9.5 % (20 ft/s 기준). 최적 원뿔각 25~30°",
    ),
    "hemispherical": CanopyData(
        "반구형", (0.62, 0.77), 1.6, (10, 15), 0.66, True, False,
        "표면적 기준 값. 투영면적 기준이면 1.4~1.5 에 해당",
    ),
    "extended_skirt": CanopyData(
        "10 % 확장 스커트", (0.78, 0.87), 1.4, (10, 15), 0.68, True, False,
        "개방하중 낮음. 제작 난이도 중간",
    ),
    "cross": CanopyData(
        "십자형", (0.60, 0.85), 1.2, (0, 3), 0.69, False, False,
        "가장 안정. 팔 비율 W/L 0.26~0.33 권장, 낮을수록 안정하지만 Cd·Cx 감소",
    ),
}
assert tuple(CANOPY_TABLE) == CANOPY_SHAPES

# Fig 5-20: 스커트 제한 없는 캐노피(평면/원뿔/십자)의 현수선비 이득
_LINE_RATIO_PTS = (1.0, 1.5, 2.0)
_LINE_GAIN_PTS = (1.00, 1.07, 1.10)
# Fig 5-19 (1 m 모델) 근사: 1.0 아래에서는 빠르게 감소, 0.6 에서 약 -15 %
_SHORT_LINE_SLOPE = 0.375
# Fig 5-21: 캔~캐노피 거리 (캔 지름 배수) 별 후류 항력 손실
_WAKE_DIST_PTS = (5.0, 10.0, 20.0)
_WAKE_FACTOR_PTS = (0.85, 0.92, 1.00)
# Fig 5-18 평면 원형: 하강속도별 Cd0 를 9 m/s 이상 값 0.80 으로 정규화
_VEL_PTS = (3.05, 4.57, 6.10, 7.62, 9.15)
_VEL_FACTOR_PTS = (1.12 / 0.80, 0.94 / 0.80, 0.85 / 0.80, 0.81 / 0.80, 1.00)


def canopy_data(shape: str) -> CanopyData:
    try:
        return CANOPY_TABLE[shape]
    except KeyError:
        raise ValueError(f"알 수 없는 캐노피 형상 {shape!r}. 가능: {CANOPY_SHAPES}") from None


def cd0_nominal(shape: str, level: str = "mid") -> float:
    """Knacke Table 5-1 범위의 하한/중앙/상한"""
    lo, hi = canopy_data(shape).cd0_range
    return {"low": lo, "mid": 0.5 * (lo + hi), "high": hi}[level]


def k_line(shape: str, line_ratio: float) -> float:
    """현수선비 Le/D0 보정 (Knacke Fig 5-19, 5-20)."""
    if line_ratio < 1.0:
        return max(0.7, 1.0 - _SHORT_LINE_SLOPE * (1.0 - line_ratio))
    r = min(line_ratio, 1.1) if canopy_data(shape).skirt_restricted else line_ratio
    return float(np.interp(r, _LINE_RATIO_PTS, _LINE_GAIN_PTS))


def k_vent(vent_ratio: float) -> float:
    """벤트 보정. 개방 면적에 비례해 항력이 준다고 근사한다."""
    if vent_ratio > 0.01:
        warnings.warn(f"벤트 면적비 {vent_ratio:.1%} 는 Knacke 권장(0.25 %) 보다 큽니다. 1 % 초과는 과다.")
    return 1.0 - vent_ratio


def k_wake(rigging_length: float, forebody_diameter: float) -> float:
    """전방동체(캔) 후류 손실 (Knacke Fig 5-21). 보수적 값이므로 리깅을 길게 잡으면 1.0 에 수렴."""
    d = rigging_length / forebody_diameter
    if d < _WAKE_DIST_PTS[0]:
        return max(0.75, _WAKE_FACTOR_PTS[0] - (_WAKE_DIST_PTS[0] - d) * 0.02)
    return float(np.interp(d, _WAKE_DIST_PTS, _WAKE_FACTOR_PTS))


def k_velocity(shape: str, descent_rate: float) -> float:
    """하강속도 효과 (Knacke Fig 5-18). 불안정 캐노피만 해당. 느릴수록 활공/진동으로 Cd 상승.

    이 이득은 흔들림·표류와 함께 오므로 설계 사이징에서는 끄고(1.0) 예측에서만 켜는 것을 권장.
    """
    if not canopy_data(shape).unstable:
        return 1.0
    return float(np.interp(descent_rate, _VEL_PTS, _VEL_FACTOR_PTS))


def opening_force_coefficient(shape: str) -> float:
    """무한질량 개방하중계수 Cx (Knacke Table 5-1)"""
    return canopy_data(shape).cx


@dataclass
class CdBreakdown:
    """Cd 계산 내역. cd_s 가 RocketPy 에 들어가는 값이다."""

    shape: str
    diameter: float
    area: float
    cd0: float
    k_line: float
    k_vent: float
    k_wake: float
    k_velocity: float
    cd: float
    source: str   # "model" | "override"
    notes: dict

    @property
    def cd_s(self) -> float:
        return self.cd * self.area

    def describe(self) -> str:
        data = canopy_data(self.shape)
        n = self.notes
        head = f"[Cd 모델] {data.label}  D0 = {self.diameter:.2f} m, S0 = {self.area:.3f} m^2"
        if self.source == "override":
            return head + f"\n  낙하시험 측정값 사용: Cd = {self.cd:.3f} (S0 기준)  ->  CdS = {self.cd_s:.3f} m^2"
        lo, hi = data.cd0_range
        vel = (f"하강속도 {n['descent_rate']:.2f} m/s 효과" if n.get("descent_rate") is not None
               else "하강속도 효과 (설계용: 끔)")
        lines = [
            head,
            f"  Cd0    Knacke Table 5-1 {lo:.2f}~{hi:.2f} 의 {n['cd_level']}",
            f"  k_line Le/D0 = {n['line_ratio']:.2f} (Le = {n['line_length']:.2f} m)",
            f"  k_vent Av/S0 = {n['vent_ratio']:.2%}",
            f"  k_wake 리깅 {n['rigging_length']:.2f} m = 캔 지름 x {n['rigging_diameters']:.1f}",
            f"  k_vel  {vel}",
        ]
        vals = [self.cd0, self.k_line, self.k_vent, self.k_wake, self.k_velocity]
        width = max(len(s) for s in lines[1:])
        out = [lines[0]] + [f"{s:<{width}} : {v:.3f}" for s, v in zip(lines[1:], vals)]
        out.append(f"  => Cd = {self.cd:.3f},  CdS = {self.cd_s:.3f} m^2"
                   f"   (Cx = {data.cx}, 진동 ±{data.oscillation_deg[0]}~{data.oscillation_deg[1]}°)")
        out.append(f"  참고: {data.note}")
        return "\n".join(out)


def design_cd(canopy: Canopy, cansat: CanSat, descent_rate: float | None = None,
              velocity_effect: bool = False) -> CdBreakdown:
    """설계 파라미터로부터 Cd 와 그 내역을 계산한다."""
    rigging = canopy.effective_rigging_length
    notes = dict(
        cd_level=canopy.cd_level, line_ratio=canopy.line_ratio, line_length=canopy.line_length,
        vent_ratio=canopy.vent_ratio, rigging_length=rigging, rigging_diameters=rigging / cansat.diameter,
        descent_rate=descent_rate if velocity_effect else None,
    )
    if canopy.cd_override is not None:
        cd = canopy.cd_override
        return CdBreakdown(canopy.shape, canopy.diameter, canopy.area, cd, 1.0, 1.0, 1.0, 1.0, cd, "override", notes)
    if velocity_effect and descent_rate is None:
        raise ValueError("velocity_effect=True 이면 descent_rate 가 필요합니다. predict_descent_rate() 를 쓰세요.")
    cd0 = cd0_nominal(canopy.shape, canopy.cd_level)
    kl = k_line(canopy.shape, canopy.line_ratio)
    kv = k_vent(canopy.vent_ratio)
    kw = k_wake(rigging, cansat.diameter)
    kvel = k_velocity(canopy.shape, descent_rate) if velocity_effect else 1.0
    cd = cd0 * kl * kv * kw * kvel
    return CdBreakdown(canopy.shape, canopy.diameter, canopy.area, cd0, kl, kv, kw, kvel, cd, "model", notes)


def predict_descent_rate(canopy: Canopy, cansat: CanSat, rho: float, velocity_effect: bool = False,
                         g: float = G0, tol: float = 1e-4, max_iter: int = 50) -> tuple[float, CdBreakdown]:
    """정상 하강속도 예측. velocity_effect=True 면 Cd(v) 와 v 를 반복 계산으로 맞춘다."""
    bd = design_cd(canopy, cansat)
    v = terminal_velocity(cansat.mass, bd.cd_s, rho, g)
    if not velocity_effect or bd.source == "override":
        return v, bd
    for _ in range(max_iter):
        bd = design_cd(canopy, cansat, descent_rate=v, velocity_effect=True)
        v_new = terminal_velocity(cansat.mass, bd.cd_s, rho, g)
        if abs(v_new - v) < tol:
            return v_new, bd
        v = 0.5 * (v + v_new)
    return v, bd


def size_for_descent_rate(shape: str, cansat: CanSat, v_target: float, rho: float, *,
                          line_ratio: float = 1.0, vent_ratio: float = 0.0025, riser_length: float = 0.30,
                          rigging_length: float | None = None, cd_level: str = "mid",
                          velocity_effect: bool = False, g: float = G0,
                          tol: float = 1e-4, max_iter: int = 50) -> tuple[float, CdBreakdown]:
    """목표 하강속도를 내는 공칭 지름 D0. Cd 가 리깅 길이를 통해 D0 에 약하게 의존하므로 반복 계산한다."""
    cd_s_req = required_cd_s(cansat.mass, v_target, rho, g)

    def make(d0: float) -> Canopy:
        return Canopy(shape=shape, diameter=d0, line_ratio=line_ratio, vent_ratio=vent_ratio,
                      riser_length=riser_length, rigging_length=rigging_length, cd_level=cd_level)

    d0 = math.sqrt(4 * cd_s_req / (math.pi * cd0_nominal(shape, cd_level)))
    for _ in range(max_iter):
        bd = design_cd(make(d0), cansat, descent_rate=v_target, velocity_effect=velocity_effect)
        d0_new = math.sqrt(4 * cd_s_req / (math.pi * bd.cd))
        if abs(d0_new - d0) < tol:
            d0 = d0_new
            break
        d0 = d0_new
    return d0, design_cd(make(d0), cansat, descent_rate=v_target, velocity_effect=velocity_effect)


def opening_load_bound(canopy: Canopy, cansat: CanSat, rho: float, v_open: float,
                       cd_s: float | None = None) -> tuple[float, float]:
    """개방 순간 하중. (정상 항력 q*CdS, 무한질량 상한 q*CdS*Cx) 를 돌려준다 [N].

    캔위성처럼 가벼운 하중은 팽창 중에 이미 감속되므로(유한질량 효과) 실제 하중은 상한보다 작다.
    """
    if cd_s is None:
        cd_s = design_cd(canopy, cansat).cd_s
    q = 0.5 * rho * v_open**2
    return q * cd_s, q * cd_s * opening_force_coefficient(canopy.shape)


# ------------------------------------------------------------------ 질량 / 수납 부피 (설계 비용 항)
@dataclass(frozen=True)
class MassModel:
    """낙하산 조립체 질량 가정. 값은 전형적인 캔위성 제작 재료 기준이며 실측으로 교체할 것."""

    fabric_gsm: float = 40.0     # g/m^2, 립스탑 나일론 (1.1 oz ≈ 37, 1.9 oz ≈ 64)
    seam_factor: float = 1.15    # 시접·테두리 보강·벤트 링 여유
    n_lines: int = 8             # 현수선 개수
    line_gpm: float = 1.2        # g/m, 얇은 케블라/다크론 라인 (550 파라코드는 약 7 g/m)
    riser_gpm: float = 3.0       # g/m
    hardware_g: float = 5.0      # 스위블·링·연결부
    pack_density: float = 0.45   # g/cm^3, 수납 밀도 (Knacke: 0.4~0.65)


def canopy_mass(canopy: Canopy, model: MassModel = MassModel()) -> float:
    """낙하산 조립체 질량 [g] = 캐노피 천 + 현수선 + 리저 + 하드웨어. 현수선 길이 Le = line_ratio × D0 가 여기서 비용이 된다."""
    fabric = model.fabric_gsm * canopy.area * model.seam_factor
    lines = model.n_lines * canopy.line_length * model.line_gpm
    riser = canopy.riser_length * model.riser_gpm
    return fabric + lines + riser + model.hardware_g


def packed_volume(canopy: Canopy, model: MassModel = MassModel()) -> float:
    """수납 부피 추정 [cm^3]. 발사관 내부(Ø110 × 200 mm ≈ 1900 cm^3)에서 캔을 뺀 공간과 비교할 것."""
    return canopy_mass(canopy, model) / model.pack_density
