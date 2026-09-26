"""그리드 서치: 설계 변수 격자 (x = D0, y = Le/D0) 전체에서 하강 시뮬을 돌려 지표 표면을 그린다.

지표 (z 축 후보):
  energy        착지 충격 에너지 ½mv² [J]              (명목, 최소가 좋음)
  drift         시작점(분리 지점) 기준 표류 거리 [m]     (명목, 최소)
  time          낙하산 하강(임무) 시간 [s]              (명목, 최대)
  margin        착지점의 부지 경계 여유 [m], 안쪽 +      (명목, 최대)
  gust_drift    강풍(10 m 풍속 × gust) 표류 [m]          (최소)
  gust_margin   강풍 시 부지 경계 여유 [m]               (최대)
  opening_load  개방 하중, 무한질량 상한 q·CdS·Cx [N]    (최소)
  mass          낙하산 조립체 질량 [g] (천 + 줄 + 리저)   (최소, Le/D0 가 크게 작용)
  cds           모델 CdS [m^2]                          (참고)

사용 (저장소 루트에서):
  python -m optimization.grid_search.sweep                          # 십자형, 9개 지표 전부
  python -m optimization.grid_search.sweep --shape hemispherical --z energy drift mass
  python -m optimization.grid_search.sweep --d0 0.5 1.2 15 --line-ratio 1.0 1.5 6 --gust 1.67
출력: output/grid_search_<shape>.png (히트맵), output/grid_search_<shape>_3d.png (3D 표면), output/grid_search_<shape>.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from cansat_parachute import (
    CANOPY_SHAPES, Canopy, build_environment, canopy_mass, design_cd, korea_cansat, opening_load_bound,
    run_descent, util,
)

ROOT = Path(__file__).resolve().parents[2]

# name: (title, unit, better, cmap)   better: "min" | "max" | None
METRICS = {
    "energy":       ("Landing impact energy", "J", "min", "viridis"),
    "drift":        ("Drift from release point", "m", "min", "magma"),
    "time":         ("Descent (mission) time", "s", "max", "cividis"),
    "margin":       ("Site boundary margin (inside +)", "m", "max", "RdYlGn"),
    "gust_drift":   ("Drift in strong wind", "m", "min", "magma"),
    "gust_margin":  ("Site boundary margin in strong wind", "m", "max", "RdYlGn"),
    "opening_load": ("Opening load (infinite-mass bound)", "N", "min", "inferno"),
    "mass":         ("Parachute assembly mass", "g", "min", "plasma"),
    "cds":          ("Model CdS", "m²", None, "Blues"),
}


def sweep(shape: str, d0s: np.ndarray, ratios: np.ndarray, gust: float):
    cansat, deploy, site = korea_cansat()
    site_gust = replace(site, gust_factor=gust)
    env, env_gust = build_environment(site), build_environment(site_gust)

    polygons = []
    if site.geometry_file:
        path = Path(site.geometry_file)
        path = path if path.is_absolute() else ROOT / path
        if path.exists():
            polygons = util.aerodrome_polygons_local(util.load_site_geometry(path), site.latitude, site.longitude)

    grid = {k: np.full((len(ratios), len(d0s)), np.nan) for k in METRICS}   # [y, x]
    grid["v_land"] = np.full((len(ratios), len(d0s)), np.nan)
    t0 = time.perf_counter()
    n_sims = 0
    for j, lr in enumerate(ratios):
        for i, d0 in enumerate(d0s):
            canopy = Canopy(shape=shape, diameter=float(d0), line_ratio=float(lr))
            cd = design_cd(canopy, cansat)
            res = run_descent(cansat, canopy, deploy, site, cd_s=cd.cd_s, env=env)
            res_g = run_descent(cansat, canopy, deploy, site_gust, cd_s=cd.cd_s, env=env_gust)
            n_sims += 2
            land = util.landing_impact(cansat.mass, res.impact_velocity, res.horizontal_impact_speed,
                                       cansat.stopping_distance)
            grid["energy"][j, i] = land.energy
            grid["v_land"][j, i] = land.v_vertical
            grid["drift"][j, i] = res.drift
            grid["time"][j, i] = res.descent_time if res.opened else 0.0
            grid["margin"][j, i] = util.signed_margin(*res.landing_xy_from_pad, polygons)
            grid["gust_drift"][j, i] = res_g.drift
            grid["gust_margin"][j, i] = util.signed_margin(*res_g.landing_xy_from_pad, polygons)
            if res.opened:
                rho_open = float(env.density(res.h_open + env.elevation))
                grid["opening_load"][j, i] = opening_load_bound(canopy, cansat, rho_open, res.v_open, cd_s=cd.cd_s)[1]
            grid["mass"][j, i] = canopy_mass(canopy)
            grid["cds"][j, i] = cd.cd_s
    print(f"{n_sims} 회 시뮬, {time.perf_counter()-t0:.1f} s  (명목 + 강풍 {site.wind_speed_10m*gust:g} m/s)")
    return grid, cansat, site, polygons


def _best(z: np.ndarray, better: str | None):
    if better is None or np.all(np.isnan(z)):
        return None
    idx = np.nanargmin(z) if better == "min" else np.nanargmax(z)
    return np.unravel_index(idx, z.shape)


def plot_heatmaps(shape, d0s, ratios, grid, names, out_path: Path):
    import matplotlib.pyplot as plt

    X, Y = np.meshgrid(d0s, ratios)
    n = len(names)
    cols = 3 if n > 4 else min(n, 2)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.4 * cols, 4.3 * rows), squeeze=False)
    for ax, name in zip(axes.flat, names):
        title, unit, better, cmap = METRICS[name]
        Z = grid[name]
        kw = {}
        if name.endswith("margin"):   # 0 을 경계로 발산 색상
            lim = float(np.nanmax(np.abs(Z))) or 1.0
            kw = dict(vmin=-lim, vmax=lim)
        im = ax.pcolormesh(X, Y, Z, cmap=cmap, shading="auto", **kw)
        try:
            cs = ax.contour(X, Y, Z, levels=8, colors="white" if not name.endswith("margin") else "black", linewidths=0.7)
            ax.clabel(cs, fmt=f"%.0f {unit}" if unit != "m²" else "%.2f", fontsize=7)
        except Exception:
            pass
        if name.endswith("margin"):
            ax.contour(X, Y, Z, levels=[0.0], colors="black", linewidths=2)
        b = _best(Z, better)
        if b is not None:
            j, i = b
            ax.plot(d0s[i], ratios[j], "r*", ms=12,
                    label=f"{better} {Z[j, i]:.0f} {unit} @ D0 {d0s[i]:.2f}, Le/D0 {ratios[j]:.2f}")
            ax.legend(loc="upper right", fontsize=7)
        ax.set(xlabel="x: D0 [m]", ylabel="y: Le/D0", title=title)
        fig.colorbar(im, ax=ax, label=unit)
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle(f"Grid search  ({shape}, nominal: Goheung 350 m, wind 3 m/s SSW)", y=1.0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_surfaces(shape, d0s, ratios, grid, names, out_path: Path):
    import matplotlib.pyplot as plt

    X, Y = np.meshgrid(d0s, ratios)
    n = len(names)
    cols = 3 if n > 4 else min(n, 2)
    rows = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(5.6 * cols, 4.8 * rows))
    for k, name in enumerate(names, start=1):
        title, unit, better, cmap = METRICS[name]
        Z = grid[name]
        ax = fig.add_subplot(rows, cols, k, projection="3d")
        surf = ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor="none", alpha=0.95)
        b = _best(Z, better)
        if b is not None:
            j, i = b
            ax.scatter([d0s[i]], [ratios[j]], [Z[j, i]], color="red", s=40, depthshade=False)
        ax.set(xlabel="D0 [m]", ylabel="Le/D0", zlabel=f"{unit}", title=title)
        ax.view_init(elev=28, azim=-135)
        fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.12)
    fig.suptitle(f"Grid search surfaces  ({shape})", y=0.995)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    p = argparse.ArgumentParser(description="설계 파라미터 그리드 서치 (D0 × Le/D0 → 여러 지표)")
    p.add_argument("--shape", choices=CANOPY_SHAPES, default="cross")
    p.add_argument("--d0", type=float, nargs=3, default=(0.4, 1.4, 21), metavar=("MIN", "MAX", "N"))
    p.add_argument("--line-ratio", type=float, nargs=3, default=(1.0, 2.0, 11), metavar=("MIN", "MAX", "N"))
    p.add_argument("--z", nargs="+", choices=list(METRICS), default=list(METRICS), help="그릴 지표 (기본 전부)")
    p.add_argument("--gust", type=float, default=2.0, help="강풍 케이스 배율 (기본 2.0 → 6 m/s)")
    p.add_argument("--no-3d", action="store_true")
    p.add_argument("--out", type=Path, default=Path("output"))
    a = p.parse_args()

    d0s = np.linspace(a.d0[0], a.d0[1], int(a.d0[2]))
    ratios = np.linspace(a.line_ratio[0], a.line_ratio[1], int(a.line_ratio[2]))
    grid, cansat, site, polygons = sweep(a.shape, d0s, ratios, a.gust)
    if not polygons:
        print("(부지 경계 데이터 없음: margin 지표는 nan)")

    csv_path = a.out / f"grid_search_{a.shape}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["v_land"] + list(METRICS)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["d0_m", "line_ratio"] + keys)
        for j, lr in enumerate(ratios):
            for i, d0 in enumerate(d0s):
                w.writerow([f"{d0:.4f}", f"{lr:.4f}"] + [f"{grid[k][j, i]:.4f}" for k in keys])

    print(f"{a.shape}: m = {cansat.mass*1000:.0f} g, 명목 바람 {site.wind_speed_10m:g} m/s from {site.wind_from_deg:g}°, "
          f"격자 {len(d0s)} × {len(ratios)}")
    print(f"  {'지표':13s} {'범위':>22s}   최적점 (better)")
    for name in a.z:
        title, unit, better, _ = METRICS[name]
        Z = grid[name]
        line = f"  {name:13s} {np.nanmin(Z):9.1f} ~ {np.nanmax(Z):9.1f} {unit:3s}"
        b = _best(Z, better)
        if b is not None:
            j, i = b
            line += f"   {better} @ D0 {d0s[i]:.2f} m, Le/D0 {ratios[j]:.2f}"
        print(line)
    png = plot_heatmaps(a.shape, d0s, ratios, grid, a.z, a.out / f"grid_search_{a.shape}.png")
    print(f"저장: {png}")
    if not a.no_3d:
        png3 = plot_surfaces(a.shape, d0s, ratios, grid, a.z, a.out / f"grid_search_{a.shape}_3d.png")
        print(f"      {png3}")
    print(f"      {csv_path}")


if __name__ == "__main__":
    main()
