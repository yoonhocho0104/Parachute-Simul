"""RocketPy 기반 하강 시뮬레이션.

로켓은 더미다. 캔위성이 로켓에서 분리되는 순간(정점 고도·초기 속도)부터 시작하고, 기체는 캔 제원의
단순 물체 + 낙하산으로만 모델링한다. 전개 후에는 RocketPy 의 3-DOF 점질량 모델
(낙하산 항력 + 부가질량 + 바람 + 중력)로 적분된다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from rocketpy import Environment, Flight, Rocket

from .parachute_model import design_cd
from .parachute_parameter import CanSat, Canopy, Deployment, Site
from .util import atmosphere_profile, bearing_deg, heading_components, power_law_wind_profile


class _DropCoincidentPhaseWarning(logging.Filter):
    """분리 즉시 전개(t=0 트리거) 시 RocketPy 가 내는 무해한 경고를 숨긴다. 단일 낙하산이므로 문제 없음."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "starting *together*" not in record.getMessage()


logging.getLogger(Flight.__module__).addFilter(_DropCoincidentPhaseWarning())

# 노즈가 아래를 향한 자세 (x축 180° 회전). RocketPy 6-DOF 는 항력을 항상 기체축 반대 방향으로만 걸기
# 때문에 꼬리부터 떨어뜨리면(기본 자세) 항력이 추력처럼 작용해 속도가 발산한다.
NOSE_DOWN_QUATERNION = (0.0, 1.0, 0.0, 0.0)


def build_environment(site: Site) -> Environment:
    """지표 기온·기압·습도로 만든 대기 프로파일 + 멱법칙 바람 프로파일을 가진 RocketPy Environment."""
    env = Environment(latitude=site.latitude, longitude=site.longitude, elevation=site.elevation)
    temperature, pressure = atmosphere_profile(
        site.surface_temperature_c, site.surface_pressure_hpa * 100.0, site.relative_humidity, site.elevation
    )
    wind_u, wind_v = power_law_wind_profile(
        site.wind_speed_10m, site.wind_from_deg, site.elevation,
        alpha=site.wind_shear_exponent, gust_factor=site.gust_factor,
    )
    # 실제 예보를 쓰려면: env.set_date((yyyy, mm, dd, hh)); env.set_atmospheric_model(type="Forecast", file="GFS")
    env.set_atmospheric_model(
        type="custom_atmosphere", pressure=pressure, temperature=temperature, wind_u=wind_u, wind_v=wind_v
    )
    return env


def build_body(cansat: CanSat) -> Rocket:
    """캔위성을 RocketPy 의 Rocket 으로 표현한 더미 기체 (모터/핀/노즈 없음)."""
    r, h, m = cansat.diameter / 2, cansat.height, cansat.mass
    i_perp = m * (3 * r**2 + h**2) / 12   # 균질 원기둥, 옆으로 구르는 축
    i_axial = m * r**2 / 2                # 세로축
    return Rocket(
        radius=r,
        mass=m,
        inertia=(i_perp, i_perp, i_axial),
        power_off_drag=cansat.cd_body,
        power_on_drag=cansat.cd_body,
        center_of_mass_without_motor=0.0,
        coordinate_system_orientation="tail_to_nose",
    )


def release_heading(deploy: Deployment, site: Site) -> float:
    """분리 시 수평 진행 방향 [deg]. 기본값은 맞바람 쪽(바람이 불어오는 방향)."""
    return site.wind_from_deg if deploy.horizontal_heading_deg is None else deploy.horizontal_heading_deg


def initial_velocity(deploy: Deployment, site: Site) -> tuple[float, float, float]:
    """분리 시 초기 속도 (vx 동, vy 북, vz 상)."""
    vx, vy = heading_components(deploy.horizontal_speed, release_heading(deploy, site))
    return vx, vy, deploy.vz0


def release_offset(deploy: Deployment, site: Site) -> tuple[float, float]:
    """발사 지점 기준 분리 지점의 수평 위치 (동, 북) [m]. 로켓이 맞바람 쪽으로 흘러간 만큼."""
    return heading_components(deploy.apogee_offset_m, release_heading(deploy, site))


@dataclass
class DescentResult:
    env: Environment
    flight: Flight
    cd_s: float
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    vz: np.ndarray
    opened: bool
    t_trigger: float | None
    t_open: float | None
    h_open: float | None       # m AGL, 완전 개방 시 고도
    v_open: float | None       # m/s, 완전 개방 시 속력
    impact_velocity: float     # m/s, 수직, 음수(아래)
    vx_impact: float           # m/s, 착지 시 수평 속도 (동)
    vy_impact: float           # m/s, 착지 시 수평 속도 (북)
    x_impact: float            # m, 분리 지점 기준
    y_impact: float
    t_final: float
    release_xy: tuple = (0.0, 0.0)   # m, 발사 지점 기준 분리 지점

    @property
    def horizontal_impact_speed(self) -> float:
        return float(np.hypot(self.vx_impact, self.vy_impact))

    @property
    def landing_xy_from_pad(self) -> tuple[float, float]:
        return (self.release_xy[0] + self.x_impact, self.release_xy[1] + self.y_impact)

    @property
    def agl(self) -> np.ndarray:
        return self.z - self.env.elevation

    @property
    def descent_time(self) -> float | None:
        return self.t_final - self.t_open if self.opened else None

    @property
    def drift(self) -> float:
        return float(np.hypot(self.x_impact, self.y_impact))

    @property
    def bearing(self) -> float:
        return bearing_deg(self.x_impact, self.y_impact)


def run_descent(cansat: CanSat, canopy: Canopy, deploy: Deployment, site: Site, *,
                cd_s: float | None = None, env: Environment | None = None,
                max_time: float = 1200.0) -> DescentResult:
    """분리 순간부터 착지까지 시뮬레이션한다. cd_s 를 주지 않으면 parachute_model 로 계산한다."""
    env = env or build_environment(site)
    body = build_body(cansat)
    if cd_s is None:
        cd_s = design_cd(canopy, cansat).cd_s

    if deploy.trigger_altitude_agl is None:
        trigger = lambda p, h, y: True  # noqa: E731  분리 즉시 전개 (lag 만큼 지연)
    else:
        trigger = deploy.trigger_altitude_agl  # 해당 고도(AGL) 하강 통과 시 전개
    body.add_parachute("Main", cd_s=cd_s, trigger=trigger, sampling_rate=100, lag=deploy.lag, noise=(0, 0, 0))

    # 상태벡터: [t, x, y, z(ASL), vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]
    vx0, vy0, vz0 = initial_velocity(deploy, site)
    z0 = env.elevation + deploy.altitude_agl
    initial = [0.0, 0.0, 0.0, z0, vx0, vy0, vz0, *NOSE_DOWN_QUATERNION, 0.0, 0.0, 0.0]
    flight = Flight(rocket=body, environment=env, rail_length=1.0, initial_solution=initial, max_time=max_time)

    sol = np.array(flight.solution)
    t, x, y, z, vx, vy, vz = sol[:, :7].T
    t_trigger = t_open = h_open = v_open = None
    opened = False
    if flight.parachute_events:
        t_trigger, chute = flight.parachute_events[0]
        t_open = t_trigger + chute.lag
        if t_open < flight.t_final:
            opened = True
            h_open = float(np.interp(t_open, t, z)) - env.elevation
            v_open = float(np.sqrt(np.interp(t_open, t, vx) ** 2 + np.interp(t_open, t, vy) ** 2
                                   + np.interp(t_open, t, vz) ** 2))

    return DescentResult(
        env=env, flight=flight, cd_s=cd_s, t=t, x=x, y=y, z=z, vx=vx, vy=vy, vz=vz,
        opened=opened, t_trigger=t_trigger, t_open=t_open, h_open=h_open, v_open=v_open,
        impact_velocity=float(flight.impact_velocity), vx_impact=float(vx[-1]), vy_impact=float(vy[-1]),
        x_impact=float(flight.x_impact), y_impact=float(flight.y_impact), t_final=float(flight.t_final),
        release_xy=release_offset(deploy, site),
    )
