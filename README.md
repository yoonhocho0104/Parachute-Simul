# Parachute-Simul

캔위성(CanSat) 낙하산 설계용 하강 시뮬레이션. 로켓은 더미로 두고, 로켓에서 분리된 이후의 낙하산 하강만 다룬다.
물리 엔진은 [RocketPy](https://docs.rocketpy.org/), 항력계수 모델은 Knacke 설계 매뉴얼 기반이다.

## 환경 설정 (conda)

```bash
conda env create -f environment.yml   # 최초 1회
conda activate rocketpy
```

노트북에서는 커널 `Python (rocketpy)` 를 선택하면 된다.

## 실행

```bash
python main.py                                   # 기준 시나리오 (국내 대회 창작부), 십자형 D0 0.75 m
python main.py --gust 2                          # 강풍 케이스 (10 m 풍속 6 m/s)
python main.py --scenario korea_high             # 슬기부: 250 m 분리, 0.75 kg
python main.py --shape hemispherical --diameter 0.9 --line-ratio 1.2
python main.py --cd 0.72                         # 낙하시험으로 잰 Cd 로 모델 덮어쓰기
python main.py --stopping-distance 5             # 착지 완충 거리 5 cm 로 충격력 계산
python main.py --apogee-offset 100               # 분리 지점이 발사 지점에서 맞바람 쪽 100 m
python main.py --target-speed 6                  # 목표 하강속도에 필요한 D0
python main.py --help                            # 전체 옵션
```

출력: Cd 계산 내역, 정상 하강속도(모델)와 착지 속도(시뮬), 하강 시간, 개방 하중, 표류 거리·방위,
착지 충격(충격량·에너지·충격력), 부지 경계 안/밖 판정과 해안선까지 거리, 지름별 하강속도·충격력 표,
그리고 4분할 그래프 `output/cansat_descent.png` (고도, 하강속도, 부지·해안선 위 지상 궤적, 완충 거리별 착지 충격력).

## 착지 충격

시뮬에서 바로 나오는 것은 착지 속도(수직·수평)까지다. 충격력은 완충 거리 d(캔 변형 + 완충재 + 지면 침하)를
가정해야 나온다. 수직 속도 v 가 d 안에서 등감속으로 0 이 된다고 보고

- 충격량 m·v [N·s], 착지 에너지 ½mv² [J]
- 접촉 시간 2d/v, 평균 충격력 m·v²/(2d), 피크 충격력 (π/2)×평균 (반정현파 펄스 가정)

기본 d = 3 cm. 콘크리트에 단단한 캔 0.5~1 cm, 잔디·흙 2~5 cm, 폼 완충재 3~8 cm 가 대략의 범위이고,
그래프 네 번째 패널이 d 에 따른 충격력 곡선이다. 기준 시나리오(0.9 kg, 7.1 m/s)에서 d = 3 cm 면 평균 약 770 N(87 g), 10 cm 면 230 N(26 g).

## 부지·해안선 지도

`data/goheung_site.geojson` 에 고흥항공센터 비행장 경계 폴리곤, 활주로 2본, 주변 해안선이 들어 있다
(© OpenStreetMap contributors, ODbL 1.0). 발사 지점 기본값은 비행장 경계 폴리곤의 도심(34.60795 N, 127.21046 E)이며
`--pad-lat/--pad-lon` 으로 바꿀 수 있다. 착지점이 경계 안인지, 경계선과 해안선까지 몇 m 인지 리포트에 찍힌다.
가장 가까운 바다는 비행장 중심에서 서쪽으로 약 3.4 km 다. 데이터 갱신은 `python scripts/fetch_site_geometry.py`.

## 코드 구조

| 모듈 | 역할 |
|---|---|
| `cansat_parachute/parachute_parameter.py` | 설계 입력 dataclass (CanSat, Canopy, Deployment, Site) 와 기준 시나리오 |
| `cansat_parachute/parachute_model.py` | 설계 파라미터 → Cd/CdS 모델, 사이징, 개방하중 |
| `cansat_parachute/simul.py` | RocketPy 환경·더미 기체·하강 비행, 결과 객체 |
| `cansat_parachute/util.py` | 해석식, 대기·바람 프로파일, 그래프 |
| `main.py` | CLI |
| `optimization/grid_search/sweep.py` | 그리드 서치: D0 × Le/D0 격자 → 9개 지표 히트맵 + 3D 표면. `python -m optimization.grid_search.sweep` |
| `scripts/fetch_site_geometry.py` | OpenStreetMap 에서 부지 경계·활주로·해안선 갱신 |
| `data/goheung_site.geojson` | 위 지형 데이터 (© OpenStreetMap contributors, ODbL) |

## 기준 시나리오 (고정값)

국내 **캔위성 체험·경연대회**(우주항공청·KAIST 인공위성연구소) 창작부 기준. `--scenario korea_high` 는 슬기부, `esa` 는 ESA 조건.

| 항목 | 값 | 근거 |
|---|---|---|
| 분리 고도 | 350 m (슬기부 250 m) | 대회규정 예정 발사고도. 공고문은 300~500 m |
| 캔위성 | 900 g, Ø83 × 115 mm | 규정 상한 창작부 1.0 kg / 슬기부 0.8 kg, 캔위성 용기 Ø83 × 115 mm |
| 전개 | 분리 즉시, 지연 1.0 s | 규정: 로켓 분리 후 즉시 전개. 지연은 라인 스트레치 + 팽창 |
| 분리 시 속도 | 수평 5 m/s 맞바람 방향, 수직 0 | 정점 분리. 로켓 웨더콕 + 사출 속도 |
| 장소 | 고흥 항공센터 34.611 N, 127.206 E, 해발 1 m | 해안 간척지 |
| 대기 | 29 °C, 1008 hPa, RH 77 % → ρ ≈ 1.149 kg/m³ | 기상청 고흥 8월 평년(1991~2020): 일최고 30.1, 평균 25.7 °C, RH 77 %. 표준대기보다 6 % 희박 |
| 바람 | 10 m 풍속 3 m/s, 남남서(200°), 멱법칙 지수 0.14 | **가정값**. 평년 풍속 통계로 교체 권장. 350 m 에서 4.9 m/s |
| 강풍 케이스 | `--gust 2` (10 m 풍속 6 m/s) | 발사 가능 상한 근처 |
| 하강속도 목표 | 7 m/s (`--target-speed`) | 규정 없음. 임무 시간과 표류 거리의 절충 |

하강속도·시간 규정은 없으므로 표류 거리(해안 부지)와 임무 시간으로 설계 목표를 정한다.

## 그리드 서치

```bash
python -m optimization.grid_search.sweep                              # 십자형, D0 0.4~1.4 × Le/D0 1.0~2.0, 9개 지표
python -m optimization.grid_search.sweep --shape hemispherical --z energy drift mass
python -m optimization.grid_search.sweep --d0 0.5 1.2 15 --line-ratio 1.0 1.5 6 --gust 1.67
```

격자 점마다 명목 시나리오와 강풍(기본 6 m/s) 시나리오 두 번 시뮬해서 다음 지표를 `output/grid_search_<shape>.png`(히트맵),
`_3d.png`(3D 표면), `.csv` 로 낸다.

| z | 의미 | 좋은 방향 |
|---|---|---|
| energy | 착지 충격 에너지 ½mv² [J] | 최소 |
| drift, gust_drift | 분리 지점 기준 표류 [m], 명목 / 강풍 | 최소 |
| time | 낙하산 하강(임무) 시간 [s] | 최대 |
| margin, gust_margin | 착지점의 부지 경계 여유 [m] (안쪽 +, 검은 선이 경계) | 최대 |
| opening_load | 개방 하중 무한질량 상한 q·CdS·Cx [N] | 최소 |
| mass | 낙하산 조립체 질량 [g], 천 40 g/m² × 1.15 + 줄 8 × Le × 1.2 g/m + 리저 + 5 g | 최소 |
| cds | 모델 CdS [m²] | 참고 |

에너지·충격력·충격량·착지속도는 전부 v 의 단조함수라 같은 표면이다. Le/D0 는 질량에서만 크게 작용한다.

## Cd 모델 요약

`Cd = Cd0(형상) × k_line(Le/D0) × k_vent(Av/S0) × k_wake(리깅 거리) [× k_vel(하강속도)]`, 기준면적 S0 = πD0²/4 (벤트 포함 천 전체 면적).

| 캐노피 | Cd0 (S0 기준) | Cx | 진동 |
|---|---|---|---|
| 평면 원형 | 0.75~0.80 | 1.7 | ±10~40° |
| 원뿔형 25~30° | 0.75~0.90 | 1.8 | ±10~30° |
| 반구형 | 0.62~0.77 | 1.6 | ±10~15° |
| 10 % 확장 스커트 | 0.78~0.87 | 1.4 | ±10~15° |
| 십자형 | 0.60~0.85 | 1.1~1.2 | 0~3° |

- 현수선비: 평면·원뿔·십자형은 1.0→1.5 에서 +7 %, 1.5→2.0 에서 +3 %. 반구형·확장스커트는 1.1 이상 이득 없음
- 벤트: 0.25 % S0 권장 (지름의 5 %). 1 % 는 과다
- 후류: 캔~캐노피 거리가 캔 지름의 5배면 −15 %, 10배 −8 %, 20배 이상 무시
- 레이놀즈수 보정 없음 (캔위성 규모 Re ≈ 2×10⁵ 포함)
- 반구형 0.62~0.77 은 표면적 기준. 투영면적 기준 1.4~1.5 와 혼동 금지
- 모델 불확실성 ±10 % 수준. 최종값은 낙하시험 CdS = 2mg/(ρv²) 로 확정할 것

## RocketPy 사용 시 주의

- 공중에서 시작할 때 자세는 노즈 아래(쿼터니언 (0,1,0,0))로 둔다. RocketPy 6-DOF 는 항력을 기체축 반대 방향으로만 걸어서 꼬리부터 떨어뜨리면 속도가 발산한다.
- 습도는 가온도로 반영하므로 `env.temperature` 는 실제 기온보다 약간 높게 나온다. 밀도는 맞다.

## 참고문헌

- Knacke, T. W., *Parachute Recovery Systems Design Manual*, NWC TP 6575, 1992
- ESA, *teach with space – CanSat Parachute Design (T10)*
- KAIST 캔위성체험·경연대회 대회규정 (cansat.kaist.ac.kr)
- Apogee Rockets, *Peak of Flight* #668, Drag Coefficients of Model Rocket Parachutes
