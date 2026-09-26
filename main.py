"""캔위성 낙하산 하강 시뮬레이션 CLI.

기본 시나리오는 국내 캔위성 경연대회 창작부(고흥 항공센터, 8월, 350 m 분리)로 고정되어 있다.
--scenario 로 슬기부(250 m) 나 ESA 조건을 고를 수 있고, 개별 값은 옵션으로 덮어쓸 수 있다.

예:
  python main.py                                   # 창작부 기준 시나리오, 십자형 D0 0.75 m
  python main.py --gust 2                          # 강풍 케이스 (10 m 풍속 6 m/s)
  python main.py --scenario korea_high             # 슬기부: 250 m, 0.75 kg
  python main.py --shape hemispherical --diameter 0.9 --line-ratio 1.2
  python main.py --cd 0.72                         # 낙하시험 측정 Cd 로 모델 덮어쓰기
  python main.py --stopping-distance 5             # 착지 완충 거리 5 cm 로 충격력 계산
  python main.py --apogee-offset 100               # 분리 지점이 발사 지점에서 맞바람 쪽 100 m
  python main.py --target-speed 6                  # 목표 하강속도에 필요한 D0
  python main.py --help
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from cansat_parachute import (
    CANOPY_SHAPES, CANOPY_TABLE, SCENARIOS, Canopy, CanSat, Deployment, DescentResult, Site,
    build_environment, design_cd, initial_velocity, opening_load_bound, predict_descent_rate,
    run_descent, size_for_descent_rate, util,
)

ROOT = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description="캔위성 낙하산 하강 시뮬레이션 (RocketPy + Knacke Cd 모델)")
    p.add_argument("--scenario", choices=sorted(SCENARIOS), default="korea",
                   help="기준 시나리오: korea(창작부 350 m) | korea_high(슬기부 250 m) | esa(1000 m)")
    g = p.add_argument_group("캔위성 (미지정 시 시나리오 값)")
    g.add_argument("--mass", type=float, help="총질량 [kg]")
    g.add_argument("--stopping-distance", type=float, help="착지 완충 거리 [cm] (기본 3)")
    g = p.add_argument_group("캐노피 설계")
    g.add_argument("--shape", choices=CANOPY_SHAPES, default=Canopy.shape)
    g.add_argument("--diameter", type=float, default=Canopy.diameter, help="공칭 지름 D0 [m]")
    g.add_argument("--line-ratio", type=float, default=Canopy.line_ratio, help="현수선비 Le/D0")
    g.add_argument("--vent-ratio", type=float, default=Canopy.vent_ratio, help="벤트 면적 / S0")
    g.add_argument("--rigging", type=float, default=None, help="캔~캐노피 거리 [m] (기본: riser + Le)")
    g.add_argument("--cd", type=float, default=None, help="측정 Cd (S0 기준). 주면 모델 대신 사용")
    g.add_argument("--cd-level", choices=("low", "mid", "high"), default=Canopy.cd_level, help="표 범위 중 선택")
    g.add_argument("--velocity-effect", action="store_true", help="불안정 캐노피의 저속 Cd 상승 효과 포함")
    g = p.add_argument_group("분리 조건 (미지정 시 시나리오 값)")
    g.add_argument("--altitude", type=float, help="분리 고도 AGL [m]")
    g.add_argument("--hspeed", type=float, help="분리 시 수평 속력 [m/s]")
    g.add_argument("--heading", type=float, help="수평 진행 방향 [deg, 북 0] (기본: 맞바람 쪽)")
    g.add_argument("--vz0", type=float, help="분리 시 수직 속도 [m/s], 아래가 음수")
    g.add_argument("--lag", type=float, help="전개 지연 [s]")
    g.add_argument("--trigger-altitude", type=float, default=None, help="이 고도(AGL)까지 자유낙하 후 전개 [m]")
    g.add_argument("--apogee-offset", type=float, help="분리 지점의 발사 지점 대비 수평 거리 [m], 맞바람 쪽")
    g = p.add_argument_group("발사장/대기 (미지정 시 시나리오 값)")
    g.add_argument("--wind", type=float, help="10 m 기준 풍속 [m/s]")
    g.add_argument("--wind-from", type=float, help="풍향(불어오는 방향) [deg]")
    g.add_argument("--gust", type=float, help="바람 프로파일 배율 (강풍 케이스 2.0)")
    g.add_argument("--shear", type=float, help="멱법칙 지수 (0.11 해상, 0.14 개활지, 0.20 교외)")
    g.add_argument("--temp", type=float, help="지표 기온 [°C]")
    g.add_argument("--pressure", type=float, help="지표 기압 [hPa]")
    g.add_argument("--rh", type=float, help="상대습도 0~1")
    g.add_argument("--elevation", type=float, help="발사장 해발고도 [m]")
    g.add_argument("--pad-lat", type=float, help="발사 지점 위도")
    g.add_argument("--pad-lon", type=float, help="발사 지점 경도")
    g.add_argument("--no-map", action="store_true", help="부지/해안선 지도 생략")
    g = p.add_argument_group("출력")
    g.add_argument("--target-speed", type=float, default=7.0, help="목표 하강속도 [m/s] (필요 D0 계산)")
    g.add_argument("--out", type=Path, default=Path("output") / "cansat_descent.png")
    g.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def _override(obj, **kwargs):
    """None 이 아닌 값만 dataclass 에 덮어쓴다."""
    return replace(obj, **{k: v for k, v in kwargs.items() if v is not None})


def print_scenario(name: str, cansat: CanSat, deploy: Deployment, site: Site, rho: float) -> None:
    vx, vy, vz = initial_velocity(deploy, site)
    heading = site.wind_from_deg if deploy.horizontal_heading_deg is None else deploy.horizontal_heading_deg
    print(f"[시나리오: {name}] {site.name}  (발사 지점 {site.latitude:.5f} N, {site.longitude:.5f} E, 해발 {site.elevation:.0f} m)")
    print(f"  대기   : {site.surface_temperature_c:.0f} °C, {site.surface_pressure_hpa:.0f} hPa, RH {site.relative_humidity:.0%}"
          f"  ->  지표 밀도 {rho:.3f} kg/m^3 (표준대기 1.225 대비 {rho/1.225-1:+.1%})")
    print(f"  바람   : 10 m 풍속 {site.wind_speed_10m*site.gust_factor:.1f} m/s from {site.wind_from_deg:.0f}°, "
          f"멱법칙 지수 {site.wind_shear_exponent}"
          f"  ->  {deploy.altitude_agl:.0f} m 고도에서 {site.wind_speed_10m*site.gust_factor*(deploy.altitude_agl/10)**site.wind_shear_exponent:.1f} m/s")
    print(f"  캔위성 : {cansat.mass*1000:.0f} g, Ø{cansat.diameter*1000:.0f} × {cansat.height*1000:.0f} mm, "
          f"착지 완충 거리 {cansat.stopping_distance*100:.1f} cm")
    print(f"  분리   : {deploy.altitude_agl:.0f} m AGL, 발사 지점에서 맞바람 쪽 {deploy.apogee_offset_m:.0f} m, "
          f"수평 {deploy.horizontal_speed:.1f} m/s (방위 {heading:.0f}°), 수직 {vz:+.1f} m/s, 전개 지연 {deploy.lag:.1f} s, "
          + ("분리 즉시 전개" if deploy.trigger_altitude_agl is None else f"{deploy.trigger_altitude_agl:.0f} m 에서 전개"))


def print_report(res: DescentResult, cansat: CanSat, canopy: Canopy, v_pred: float,
                 landing: util.LandingImpact | None, site_map: util.SiteMap | None) -> None:
    line = "=" * 64
    print(line)
    print("CanSat parachute descent (RocketPy)")
    print(line)
    print(f"낙하산 CdS         : {res.cd_s:7.3f} m^2")
    if not res.opened:
        print("!! 착지 전에 낙하산이 완전히 펴지지 않았습니다. trigger_altitude / lag 를 확인하세요.")
        print(f"착지 속도          : {abs(res.impact_velocity):7.2f} m/s (자유낙하), 총 비행 {res.t_final:.1f} s")
        print(line)
        return
    rho_open = float(res.env.density(res.h_open + res.env.elevation))
    f_steady, f_bound = opening_load_bound(canopy, cansat, rho_open, res.v_open, cd_s=res.cd_s)
    print(f"전개 완료          : t = {res.t_open:6.2f} s, 고도 {res.h_open:6.1f} m AGL, 속도 {res.v_open:5.2f} m/s")
    print(f"개방 하중          : 항력 {f_steady:6.1f} N, 무한질량 상한 {f_bound:6.1f} N "
          f"(= {f_bound/(cansat.mass*util.G0):.0f} g, 실제는 이보다 작음)")
    print(f"정상 하강속도(모델): {v_pred:7.2f} m/s")
    print(f"착지 속도(시뮬)    : 수직 {abs(res.impact_velocity):5.2f} m/s, 수평 {res.horizontal_impact_speed:5.2f} m/s (바람에 실림)")
    print(f"낙하산 하강 시간   : {res.descent_time:7.1f} s   (총 비행 {res.t_final:.1f} s)")
    lx, ly = res.landing_xy_from_pad
    print(f"착지 지점          : E {lx:+8.1f} m, N {ly:+8.1f} m  (발사 지점 기준)")
    print(f"표류 거리          : {res.drift:7.1f} m  (분리 지점 기준, 방위 {res.bearing:.0f} deg)")
    if landing:
        print("-" * 64)
        print(f"착지 충격 (완충 거리 {landing.stopping_distance*100:.1f} cm 가정)")
        print(f"  충격량 m·v       : {landing.impulse:6.2f} N·s      착지 에너지 ½mv²: {landing.energy:5.1f} J")
        print(f"  접촉 시간        : {landing.contact_time*1000:6.1f} ms")
        print(f"  평균 충격력      : {landing.force_avg:6.0f} N ({landing.g_avg():.0f} g)"
              f"    피크(반정현) {landing.force_peak:6.0f} N ({landing.g_peak():.0f} g)")
        for d_cm in (1, 5, 10):
            alt = util.landing_impact(landing.mass, landing.v_vertical, landing.v_horizontal, d_cm / 100)
            print(f"  완충 {d_cm:2d} cm 이면   : 평균 {alt.force_avg:6.0f} N ({alt.g_avg():.0f} g)")
    if site_map:
        print("-" * 64)
        inside = {True: "비행장 경계 안", False: "비행장 경계 밖 !!", None: "경계 데이터 없음"}[site_map.landing_inside]
        b = f", 경계선까지 {site_map.boundary_distance:.0f} m" if site_map.boundary_distance is not None else ""
        c = (f"해안선까지 {site_map.coast_distance/1000:.2f} km (방위 {util.bearing_deg(site_map.coast_point[0]-lx, site_map.coast_point[1]-ly):.0f}°)"
             if site_map.coast_distance is not None else "해안선 데이터 없음")
        print(f"부지 판정          : {inside}{b}")
        print(f"바다               : {c}")
    print(line)


def print_sizing(cansat: CanSat, canopy: Canopy, rho: float, target_speed: float, velocity_effect: bool) -> None:
    label = CANOPY_TABLE[canopy.shape].label
    print(f"\n--- 지름 vs 정상 하강속도 ({label}, Le/D0 = {canopy.line_ratio}, m = {cansat.mass*1000:.0f} g, "
          f"rho = {rho:.3f} kg/m^3) ---")
    print("  D0 [m]    Cd     CdS [m^2]   v [m/s]   착지 평균 충격력 (완충 {:.0f} cm) [N]".format(cansat.stopping_distance * 100))
    for d in np.arange(0.4, 1.41, 0.1):
        v, bd = predict_descent_rate(replace(canopy, diameter=float(d)), cansat, rho, velocity_effect)
        f_avg = util.landing_impact(cansat.mass, v, 0.0, cansat.stopping_distance).force_avg
        print(f"  {d:5.2f}   {bd.cd:5.3f}    {bd.cd_s:6.3f}     {v:6.2f}     {f_avg:8.0f}")
    if canopy.cd_override is None:
        d0, bd = size_for_descent_rate(
            canopy.shape, cansat, target_speed, rho, line_ratio=canopy.line_ratio, vent_ratio=canopy.vent_ratio,
            riser_length=canopy.riser_length, rigging_length=canopy.rigging_length, cd_level=canopy.cd_level,
            velocity_effect=velocity_effect,
        )
    else:
        d0 = util.diameter_for_cd_s(util.required_cd_s(cansat.mass, target_speed, rho), canopy.cd_override)
        bd = design_cd(replace(canopy, diameter=d0), cansat)
    print(f"\n목표 하강속도 {target_speed:.1f} m/s 에 필요한 공칭 지름: D0 = {d0:.2f} m  (Cd = {bd.cd:.3f}, CdS = {bd.cd_s:.3f} m^2)")


def main():
    a = parse_args()
    cansat, deploy, site = SCENARIOS[a.scenario]()
    cansat = _override(cansat, mass=a.mass,
                       stopping_distance=None if a.stopping_distance is None else a.stopping_distance / 100)
    deploy = _override(deploy, altitude_agl=a.altitude, horizontal_speed=a.hspeed, horizontal_heading_deg=a.heading,
                       vz0=a.vz0, lag=a.lag, trigger_altitude_agl=a.trigger_altitude, apogee_offset_m=a.apogee_offset)
    site = _override(site, wind_speed_10m=a.wind, wind_from_deg=a.wind_from, gust_factor=a.gust,
                     wind_shear_exponent=a.shear, surface_temperature_c=a.temp, surface_pressure_hpa=a.pressure,
                     relative_humidity=a.rh, elevation=a.elevation, latitude=a.pad_lat, longitude=a.pad_lon)
    canopy = Canopy(shape=a.shape, diameter=a.diameter, line_ratio=a.line_ratio, vent_ratio=a.vent_ratio,
                    rigging_length=a.rigging, cd_override=a.cd, cd_level=a.cd_level)

    env = build_environment(site)
    rho = float(env.density(env.elevation))
    print_scenario(a.scenario, cansat, deploy, site, rho)
    print()
    v_pred, cd = predict_descent_rate(canopy, cansat, rho, velocity_effect=a.velocity_effect)
    print(cd.describe())
    print()

    res = run_descent(cansat, canopy, deploy, site, cd_s=cd.cd_s, env=env)

    landing = site_map = None
    if res.opened:
        landing = util.landing_impact(cansat.mass, res.impact_velocity, res.horizontal_impact_speed,
                                      cansat.stopping_distance)
    if site.geometry_file and not a.no_map:
        geo_path = Path(site.geometry_file)
        if not geo_path.is_absolute():
            geo_path = ROOT / geo_path
        if geo_path.exists():
            geometry = util.load_site_geometry(geo_path)
            site_map = util.build_site_map(geometry, site.latitude, site.longitude, res.release_xy,
                                           (res.x_impact, res.y_impact))
        else:
            print(f"(지형 파일 없음: {geo_path}. scripts/fetch_site_geometry.py 로 받을 수 있음)")

    print_report(res, cansat, canopy, v_pred, landing, site_map)
    print_sizing(cansat, canopy, rho, a.target_speed, a.velocity_effect)

    if res.opened and not a.no_plot:
        out = util.plot_descent(res.t, res.x, res.y, res.agl, res.vz, res.t_open, v_pred, a.out,
                                site_map=site_map, landing=landing)
        print(f"그래프 저장: {out}")


if __name__ == "__main__":
    main()
