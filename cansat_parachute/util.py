"""공용 수식과 도구. RocketPy 에 의존하지 않는다."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

G0 = 9.80665        # m/s^2
R_DRY = 287.05      # J/(kg K), 건조공기 기체상수
LAPSE_RATE = 0.0065 # K/m, 표준 기온감률


# ------------------------------------------------------------------ 낙하산 기본 수식
def terminal_velocity(mass: float, cd_s: float, rho: float, g: float = G0) -> float:
    """정상 하강속도 v = sqrt(2 m g / (rho * CdS))"""
    return math.sqrt(2 * mass * g / (rho * cd_s))


def required_cd_s(mass: float, v_target: float, rho: float, g: float = G0) -> float:
    """목표 하강속도를 내기 위한 CdS"""
    return 2 * mass * g / (rho * v_target**2)


def diameter_for_cd_s(cd_s: float, cd: float) -> float:
    """CdS 와 Cd 로부터 공칭 지름 D0"""
    return math.sqrt(4 * cd_s / (math.pi * cd))


# ------------------------------------------------------------------ 방향/바람
def wind_components(speed: float, from_deg: float) -> tuple[float, float]:
    """기상 규약 풍향(불어오는 방향) -> (동쪽 성분 u, 북쪽 성분 v). 서풍(270)이면 u = +speed."""
    rad = math.radians(from_deg)
    return -speed * math.sin(rad), -speed * math.cos(rad)


def heading_components(speed: float, heading_deg: float) -> tuple[float, float]:
    """진행 방향(북 0°, 시계방향) -> (동쪽 성분, 북쪽 성분)"""
    rad = math.radians(heading_deg)
    return speed * math.sin(rad), speed * math.cos(rad)


def bearing_deg(x_east: float, y_north: float) -> float:
    """북 기준 시계방향 방위각 [deg]"""
    return math.degrees(math.atan2(x_east, y_north)) % 360


def power_law_wind_profile(speed_ref: float, from_deg: float, elevation: float, *, z_ref: float = 10.0,
                           alpha: float = 0.14, gust_factor: float = 1.0, top: float = 3000.0,
                           step: float = 25.0) -> tuple[np.ndarray, np.ndarray]:
    """기준고도 풍속으로부터 멱법칙 바람 프로파일 u(z) = u_ref (z/z_ref)^alpha.

    alpha 0.11 해상/평탄 개활지, 0.14 개활 초지(설계 관행), 0.20 교외.
    반환: wind_u [(h_asl, u_east)], wind_v [(h_asl, v_north)]  (RocketPy custom_atmosphere 입력 형식)
    """
    z = np.arange(0.0, top + step, step)
    speed = gust_factor * speed_ref * (np.maximum(z, 1.0) / z_ref) ** alpha
    ue, vn = wind_components(1.0, from_deg)
    h = z + elevation
    return np.column_stack([h, speed * ue]), np.column_stack([h, speed * vn])


# ------------------------------------------------------------------ 대기
def saturation_vapor_pressure_pa(t_c: float) -> float:
    """Magnus 식 (Alduchov & Eskridge 1996)"""
    return 610.94 * math.exp(17.625 * t_c / (t_c + 243.04))


def virtual_temperature_k(t_c: float, p_pa: float, rh: float) -> float:
    """가온도. 습한 공기의 밀도를 건조공기 식 p/(R Tv) 로 계산하기 위한 온도"""
    e = rh * saturation_vapor_pressure_pa(t_c)
    return (t_c + 273.15) / (1.0 - 0.378 * e / p_pa)


def air_density(t_c: float, p_pa: float, rh: float = 0.0) -> float:
    return p_pa / (R_DRY * virtual_temperature_k(t_c, p_pa, rh))


def atmosphere_profile(t_surface_c: float, p_surface_pa: float, rh: float, elevation: float, *,
                       top: float = 3000.0, step: float = 50.0) -> tuple[np.ndarray, np.ndarray]:
    """지표 관측값에서 시작하는 온도/압력 프로파일. 온도는 표준 감률, 압력은 정역학 평형.

    습도는 가온도로 반영해 밀도가 맞도록 한다 (RocketPy 는 건조공기 식으로 밀도를 계산).
    반환: temperature [(h_asl, T_K)], pressure [(h_asl, p_Pa)]
    """
    tv0 = virtual_temperature_k(t_surface_c, p_surface_pa, rh)
    z = np.arange(0.0, top + step, step)
    tv = tv0 - LAPSE_RATE * z
    p = p_surface_pa * (tv / tv0) ** (G0 / (R_DRY * LAPSE_RATE))
    h = z + elevation
    return np.column_stack([h, tv]), np.column_stack([h, p])


# ------------------------------------------------------------------ 착지 충격
@dataclass(frozen=True)
class LandingImpact:
    """착지 충격. 지면에 수직인 속도 성분이 완충 거리 d 안에서 등감속으로 0 이 된다고 가정한다.

    시뮬에서 바로 나오는 것은 속도까지이고, 힘은 완충 거리(캔 변형 + 완충재 + 지면 침하)를 가정해야 나온다.
    F_avg = m v^2 / (2 d),  접촉시간 = 2 d / v,  피크는 반정현파 펄스 가정으로 (pi/2) F_avg.
    """

    mass: float
    v_vertical: float        # m/s, 양수
    v_horizontal: float      # m/s, 바람에 실린 수평 속도
    stopping_distance: float # m

    @property
    def speed(self) -> float:
        return math.hypot(self.v_vertical, self.v_horizontal)

    @property
    def impulse(self) -> float:
        """수직 충격량 m·v [N·s]"""
        return self.mass * self.v_vertical

    @property
    def energy(self) -> float:
        """착지 운동에너지 [J]"""
        return 0.5 * self.mass * self.speed**2

    @property
    def contact_time(self) -> float:
        return 2 * self.stopping_distance / self.v_vertical

    @property
    def force_avg(self) -> float:
        return self.mass * self.v_vertical**2 / (2 * self.stopping_distance)

    @property
    def force_peak(self) -> float:
        return math.pi / 2 * self.force_avg

    def g_avg(self, g: float = G0) -> float:
        return self.force_avg / (self.mass * g)

    def g_peak(self, g: float = G0) -> float:
        return self.force_peak / (self.mass * g)


def landing_impact(mass: float, v_vertical: float, v_horizontal: float = 0.0,
                   stopping_distance: float = 0.03) -> LandingImpact:
    return LandingImpact(mass, abs(v_vertical), abs(v_horizontal), stopping_distance)


def landing_force_curve(mass: float, v_vertical: float, distances) -> tuple[np.ndarray, np.ndarray]:
    """완충 거리 배열에 대한 (평균 충격력, 피크 충격력) [N]"""
    d = np.asarray(distances, dtype=float)
    f_avg = mass * v_vertical**2 / (2 * d)
    return f_avg, math.pi / 2 * f_avg


# ------------------------------------------------------------------ 지형 (부지 경계, 해안선)
EARTH_RADIUS = 6371000.0


def latlon_to_local(lat, lon, lat0: float, lon0: float):
    """위경도 -> 기준점 기준 로컬 (동 x, 북 y) [m]. 수 km 범위의 등장방형 근사."""
    k = EARTH_RADIUS * math.pi / 180.0
    x = (np.asarray(lon, dtype=float) - lon0) * math.cos(math.radians(lat0)) * k
    y = (np.asarray(lat, dtype=float) - lat0) * k
    return x, y


def load_site_geometry(path) -> dict:
    """data/*.geojson 을 읽어 kind 별로 [lon, lat] 좌표 배열 목록을 돌려준다."""
    fc = json.loads(Path(path).read_text())
    out = {"aerodrome": [], "runway": [], "coastline": [], "source": fc.get("properties", {}).get("source", "")}
    for f in fc["features"]:
        kind = f["properties"].get("kind")
        g = f["geometry"]
        coords = g["coordinates"][0] if g["type"] == "Polygon" else g["coordinates"]
        if kind in out:
            out[kind].append(np.asarray(coords, dtype=float))
    return out


def polygon_centroid(lonlat: np.ndarray) -> tuple[float, float]:
    """(lon, lat) 폴리곤의 도심 -> (lat, lon)"""
    x, y = lonlat[:, 0], lonlat[:, 1]
    cross = x[:-1] * y[1:] - x[1:] * y[:-1]
    a = cross.sum() / 2
    cx = ((x[:-1] + x[1:]) * cross).sum() / (6 * a)
    cy = ((y[:-1] + y[1:]) * cross).sum() / (6 * a)
    return float(cy), float(cx)


def point_in_polygon(px: float, py: float, poly_xy: np.ndarray) -> bool:
    """ray casting"""
    x, y = poly_xy[:, 0], poly_xy[:, 1]
    inside = False
    j = len(x) - 1
    for i in range(len(x)):
        if (y[i] > py) != (y[j] > py):
            x_cross = (x[j] - x[i]) * (py - y[i]) / (y[j] - y[i]) + x[i]
            if px < x_cross:
                inside = not inside
        j = i
    return inside


def distance_to_polyline(px: float, py: float, line_xy: np.ndarray) -> tuple[float, tuple[float, float], int]:
    """점에서 폴리라인까지 최단거리, 최근접점, 최근접 선분 인덱스"""
    a = line_xy[:-1]
    b = line_xy[1:]
    ab = b - a
    ap = np.array([px, py]) - a
    t = np.clip((ap * ab).sum(axis=1) / np.maximum((ab * ab).sum(axis=1), 1e-9), 0.0, 1.0)
    proj = a + t[:, None] * ab
    d = np.hypot(proj[:, 0] - px, proj[:, 1] - py)
    i = int(np.argmin(d))
    return float(d[i]), (float(proj[i, 0]), float(proj[i, 1])), i


def aerodrome_polygons_local(geometry: dict, lat0: float, lon0: float) -> list:
    """비행장 경계 폴리곤들을 (lat0, lon0) 기준 로컬 (동, 북) m 좌표로"""
    out = []
    for poly in geometry["aerodrome"]:
        x, y = latlon_to_local(poly[:, 1], poly[:, 0], lat0, lon0)
        out.append(np.column_stack([x, y]))
    return out


def signed_margin(x: float, y: float, polygons: list) -> float:
    """점에서 부지 경계까지 부호 거리 [m]. 안쪽 +, 바깥 -. 폴리곤 없으면 nan."""
    if not polygons:
        return math.nan
    inside = any(point_in_polygon(x, y, poly) for poly in polygons)
    dist = min(distance_to_polyline(x, y, poly)[0] for poly in polygons)
    return dist if inside else -dist


@dataclass
class SiteMap:
    """그래프용 로컬 좌표(발사 지점 원점, m) 지형과 착지 판정 결과"""

    aerodrome: list          # 폴리곤 (N,2)
    runways: list            # 선 (N,2)
    coastlines: list         # 선 (N,2)
    release_xy: tuple        # 분리 지점
    landing_xy: tuple        # 착지 지점
    landing_inside: bool | None
    boundary_distance: float | None   # 비행장 경계선까지 거리 [m]
    coast_distance: float | None      # 해안선까지 거리 [m]
    coast_point: tuple | None
    sea_label_xy: tuple | None
    source: str = ""


def build_site_map(geometry: dict, pad_lat: float, pad_lon: float, release_xy, landing_xy_from_release) -> SiteMap:
    """지형을 발사 지점 기준 로컬 좌표로 바꾸고 착지 지점을 판정한다. 착지 좌표는 분리 지점 기준 -> 발사 지점 기준으로 변환."""
    def to_xy(lonlat):
        x, y = latlon_to_local(lonlat[:, 1], lonlat[:, 0], pad_lat, pad_lon)
        return np.column_stack([x, y])

    aero = [to_xy(g) for g in geometry["aerodrome"]]
    runways = [to_xy(g) for g in geometry["runway"]]
    coasts = [to_xy(g) for g in geometry["coastline"]]
    lx = release_xy[0] + landing_xy_from_release[0]
    ly = release_xy[1] + landing_xy_from_release[1]

    inside = boundary = None
    if aero:
        inside = any(point_in_polygon(lx, ly, poly) for poly in aero)
        boundary = min(distance_to_polyline(lx, ly, poly)[0] for poly in aero)
    coast_d = coast_pt = sea_xy = None
    if coasts:
        best = min((distance_to_polyline(lx, ly, c) + (c,) for c in coasts), key=lambda r: r[0])
        coast_d, coast_pt, seg, line = best
        # OSM 해안선 규약: 진행방향 왼쪽이 육지, 오른쪽이 바다 -> 오른쪽 법선 방향에 SEA 라벨
        dx, dy = line[seg + 1] - line[seg]
        n = math.hypot(dx, dy) or 1.0
        sea_xy = (coast_pt[0] + dy / n * 400.0, coast_pt[1] - dx / n * 400.0)
    return SiteMap(aero, runways, coasts, tuple(release_xy), (lx, ly), inside, boundary, coast_d, coast_pt, sea_xy,
                   geometry.get("source", ""))


# ------------------------------------------------------------------ 그래프
def plot_descent(t, x, y, agl, vz, t_open: float, v_term: float, out_path: Path, *,
                 site_map: SiteMap | None = None, landing: LandingImpact | None = None) -> Path:
    """고도, 하강속도, 지상 궤적(부지·해안선), 착지 충격력 4분할 그래프를 PNG 로 저장한다."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    ax = axes[0, 0]
    ax.plot(t, agl)
    ax.axvline(t_open, ls=":", c="gray")
    ax.set(xlabel="time [s]", ylabel="altitude AGL [m]", title="Altitude")
    ax = axes[0, 1]
    ax.plot(t, -vz, label="sim")
    ax.axhline(v_term, ls="--", c="gray", label=f"terminal {v_term:.2f} m/s")
    ax.axvline(t_open, ls=":", c="gray")
    ax.set(xlabel="time [s]", ylabel="descent rate [m/s]", title="Descent rate")
    ax.set_ylim(0, max(v_term * 1.5, float(np.max(-vz)) * 1.05))
    ax.legend()

    # --- 지상 궤적 + 부지/해안선
    ax = axes[1, 0]
    ox, oy = (site_map.release_xy if site_map else (0.0, 0.0))
    tx, ty = np.asarray(x) + ox, np.asarray(y) + oy
    title = "Ground track"
    if site_map:
        for poly in site_map.aerodrome:
            ax.fill(poly[:, 0], poly[:, 1], color="tab:green", alpha=0.15, lw=0)
            ax.plot(poly[:, 0], poly[:, 1], color="tab:green", lw=1.5, label="aerodrome boundary")
        for rw in site_map.runways:
            ax.plot(rw[:, 0], rw[:, 1], color="0.35", lw=4, solid_capstyle="butt", label="runway")
        for c in site_map.coastlines:
            ax.plot(c[:, 0], c[:, 1], color="tab:blue", lw=2, label="coastline")
        if site_map.sea_label_xy:
            ax.annotate("SEA", site_map.sea_label_xy, color="tab:blue", fontsize=11, fontweight="bold",
                        ha="center", va="center")
        if site_map.coast_point:
            lx, ly = site_map.landing_xy
            ax.plot([lx, site_map.coast_point[0]], [ly, site_map.coast_point[1]], ls=":", color="tab:blue", lw=1)
        ax.plot(0, 0, "k^", ms=8, label="launch pad")
        inside = {True: "inside site", False: "OUTSIDE site", None: "site n/a"}[site_map.landing_inside]
        coast = f", coast {site_map.coast_distance/1000:.1f} km" if site_map.coast_distance else ""
        title = f"Ground track ({inside}{coast})"
    ax.plot(tx, ty, color="tab:red", lw=1.5)
    ax.plot(tx[0], ty[0], "o", color="gray", label="release")
    ax.plot(tx[-1], ty[-1], "rx", ms=9, mew=2, label="landing")
    ax.set(xlabel="East from pad [m]", ylabel="North from pad [m]", title=title)
    if site_map and site_map.aerodrome:
        allpts = np.vstack(site_map.aerodrome + [np.column_stack([tx, ty])])
        pad = 600.0
        if site_map.coast_distance:
            pad = min(max(pad, site_map.coast_distance + 500.0), 4500.0)
        ax.set_xlim(allpts[:, 0].min() - pad, allpts[:, 0].max() + pad)
        ax.set_ylim(allpts[:, 1].min() - pad, allpts[:, 1].max() + pad)
    ax.set_aspect("equal", adjustable="box")
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), fontsize=8, loc="upper right")
    if site_map and site_map.aerodrome:
        # 비행장 확대 인셋: 해안선까지 담은 전체 지도에서는 부지 안의 궤적이 작게 보이므로
        axins = ax.inset_axes([0.03, 0.03, 0.46, 0.46])
        for poly in site_map.aerodrome:
            axins.fill(poly[:, 0], poly[:, 1], color="tab:green", alpha=0.15, lw=0)
            axins.plot(poly[:, 0], poly[:, 1], color="tab:green", lw=1.5)
        for rw in site_map.runways:
            axins.plot(rw[:, 0], rw[:, 1], color="0.35", lw=5, solid_capstyle="butt")
        axins.plot(0, 0, "k^", ms=8)
        axins.plot(tx, ty, color="tab:red", lw=1.5)
        axins.plot(tx[0], ty[0], "o", color="gray")
        axins.plot(tx[-1], ty[-1], "rx", ms=9, mew=2)
        zoom = np.vstack(site_map.aerodrome + [np.column_stack([tx, ty])])
        m = 200.0
        axins.set_xlim(zoom[:, 0].min() - m, zoom[:, 0].max() + m)
        axins.set_ylim(zoom[:, 1].min() - m, zoom[:, 1].max() + m)
        axins.set_aspect("equal", adjustable="box")
        axins.tick_params(labelsize=7)
        axins.grid(alpha=0.3)
        axins.set_facecolor("white")
        ax.indicate_inset_zoom(axins, edgecolor="0.3")

    # --- 착지 충격력 vs 완충 거리
    ax = axes[1, 1]
    if landing:
        d = np.linspace(0.005, 0.15, 200)
        f_avg, f_peak = landing_force_curve(landing.mass, landing.v_vertical, d)
        ax.plot(d * 100, f_avg, label="average force  m v² / 2d")
        ax.plot(d * 100, f_peak, ls="--", label="peak (half-sine)  π/2 × avg")
        ax.axvspan(0.5, 1.0, color="0.5", alpha=0.15)
        ax.axvspan(2.0, 5.0, color="tab:green", alpha=0.12)
        ax.axvspan(3.0, 8.0, color="tab:orange", alpha=0.10)
        ymax = float(f_peak[d >= 0.01].max()) * 1.05
        ax.text(0.75, ymax * 0.97, "concrete", ha="center", va="top", fontsize=8, color="0.3")
        ax.text(3.5, ymax * 0.97, "grass/soil", ha="center", va="top", fontsize=8, color="darkgreen")
        ax.text(5.5, ymax * 0.90, "foam bumper", ha="center", va="top", fontsize=8, color="darkorange")
        dc = landing.stopping_distance * 100
        ax.plot([dc], [landing.force_avg], "ko")
        ax.annotate(f"d = {dc:.1f} cm\n{landing.force_avg:.0f} N ({landing.g_avg():.0f} g) avg\n"
                    f"{landing.force_peak:.0f} N ({landing.g_peak():.0f} g) peak",
                    (dc, landing.force_avg), xytext=(12, 18), textcoords="offset points", fontsize=8,
                    arrowprops=dict(arrowstyle="-", color="0.4"))
        ax.text(0.98, 0.97,
                f"impact  v_vert {landing.v_vertical:.2f} m/s, v_horiz {landing.v_horizontal:.2f} m/s\n"
                f"impulse m·v = {landing.impulse:.1f} N·s\nenergy ½mv² = {landing.energy:.1f} J\n"
                f"contact time {landing.contact_time*1000:.0f} ms at d = {dc:.1f} cm",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", ec="0.7"))
        ax.set_ylim(0, ymax)
        ax.set_xlim(0.5, 15)
        ax2 = ax.twinx()
        ax2.set_ylim(0, ymax / (landing.mass * G0))
        ax2.set_ylabel("deceleration [g]")
        ax.set(xlabel="stopping distance d [cm]", ylabel="landing impact force [N]",
               title=f"Landing impact force (m = {landing.mass*1000:.0f} g)")
        ax.legend(fontsize=8, loc="center right")
    else:
        ax.axis("off")

    for a in (axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]):
        a.grid(alpha=0.3)
    if site_map and site_map.source:
        fig.text(0.995, 0.005, f"map data: {site_map.source}", ha="right", va="bottom", fontsize=7, color="0.4")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
