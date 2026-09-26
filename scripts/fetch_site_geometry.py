"""OpenStreetMap(Overpass API)에서 발사장 주변 지형을 받아 data/*.geojson 으로 저장한다.

비행장 경계(aeroway=aerodrome), 활주로(aeroway=runway), 해안선(natural=coastline).
저작권: © OpenStreetMap contributors, ODbL 1.0. 배포 시 출처 표기 필요.

사용:  python scripts/fetch_site_geometry.py [--lat 34.611 --lon 127.206 --radius 3000 --coast-radius 5000]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import urllib.parse
import urllib.request
from pathlib import Path

OVERPASS = "https://overpass-api.de/api/interpreter"


def fetch(lat: float, lon: float, radius: float, coast_radius: float) -> dict:
    q = (f'[out:json][timeout:90];('
         f'way["aeroway"="aerodrome"](around:{radius},{lat},{lon});'
         f'way["aeroway"="runway"](around:{radius},{lat},{lon});'
         f'way["natural"="coastline"](around:{coast_radius},{lat},{lon});'
         f');out geom;')
    req = urllib.request.Request(OVERPASS, data=urllib.parse.urlencode({"data": q}).encode(),
                                 headers={"User-Agent": "Parachute-Simul/0.1"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def to_geojson(overpass: dict, name: str) -> dict:
    features = []
    for e in overpass["elements"]:
        tags = e.get("tags", {})
        coords = [[g["lon"], g["lat"]] for g in e.get("geometry", [])]
        if not coords:
            continue
        if tags.get("aeroway") == "aerodrome":
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            geom, kind = {"type": "Polygon", "coordinates": [coords]}, "aerodrome"
        elif tags.get("aeroway") == "runway":
            geom, kind = {"type": "LineString", "coordinates": coords}, "runway"
        elif tags.get("natural") == "coastline":
            geom, kind = {"type": "LineString", "coordinates": coords}, "coastline"
        else:
            continue
        props = {"kind": kind, "osm_id": e["id"],
                 **{k: v for k, v in tags.items() if k in ("name", "name:en", "ref", "length", "width")}}
        features.append({"type": "Feature", "properties": props, "geometry": geom})
    return {
        "type": "FeatureCollection",
        "properties": {
            "name": name,
            "source": f"© OpenStreetMap contributors, ODbL 1.0 (Overpass API, {dt.date.today()})",
            "note": "coastline: OSM 규약상 진행방향 왼쪽이 육지, 오른쪽이 바다",
        },
        "features": features,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lat", type=float, default=34.611)
    p.add_argument("--lon", type=float, default=127.206)
    p.add_argument("--radius", type=float, default=3000, help="비행장/활주로 검색 반경 [m]")
    p.add_argument("--coast-radius", type=float, default=5000, help="해안선 검색 반경 [m]")
    p.add_argument("--name", default="고흥항공센터 주변 지형 (비행장 경계, 활주로, 해안선)")
    p.add_argument("--out", type=Path, default=Path("data") / "goheung_site.geojson")
    a = p.parse_args()
    fc = to_geojson(fetch(a.lat, a.lon, a.radius, a.coast_radius), a.name)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(fc, ensure_ascii=False, indent=1))
    kinds = [f["properties"]["kind"] for f in fc["features"]]
    print(f"saved {a.out}: " + ", ".join(f"{k} x{kinds.count(k)}" for k in sorted(set(kinds))))


if __name__ == "__main__":
    main()
