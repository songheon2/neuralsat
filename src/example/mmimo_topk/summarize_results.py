"""
manifest.csv(idx,eps,net,data,k,abs_row,result)와 run_batch.py가 만든 result
파일들을 모아서 summary.csv(idx,eps,status,runtime)와 idx별 certified-robust
반경을 robustness_summary.csv로 정리하고, eps별 unsat/sat 비율을 floating
range 차트(I-beam 스타일)로 eps_unsat_ratio.png(Pillow만 사용, 새 의존성 없음)에
저장한다.

robust radius 정의: eps를 오름차순으로 볼 때, 가장 작은 eps부터 연속으로
unsat인 구간의 마지막 eps (= 그보다 작은 eps는 전부 unsat으로 확인된 반경).
중간에 unsat이 아닌 결과가 나온 뒤 더 큰 eps에서 다시 unsat이 나오면(비단조),
anomaly로 표시한다 (L_inf eps-ball이 커질수록 sat/unknown 쪽으로 가는게
자연스러운데 그 가정이 깨진 경우).
"""

from __future__ import annotations

import argparse
import csv
import pathlib

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_MANIFEST = SCRIPT_DIR / "vnnlib" / "manifest.csv"
DEFAULT_SUMMARY = SCRIPT_DIR / "summary.csv"
DEFAULT_ROBUSTNESS_SUMMARY = SCRIPT_DIR / "robustness_summary.csv"
DEFAULT_CHART = SCRIPT_DIR / "eps_unsat_ratio.png"
DEFAULT_CHART_7 = SCRIPT_DIR / "eps_unsat_ratio_7.png"
DEFAULT_LINE_CHART = SCRIPT_DIR / "eps_unsat_line.png"
DEFAULT_EPS_SUMMARY = SCRIPT_DIR / "eps_ratio_summary.csv"

_CHART_INK = (11, 11, 11)
_CHART_MUTED = (137, 135, 129)
_CHART_BASELINE = (195, 194, 183)
_CHART_GRID = (225, 224, 217)
_LINE_UNSAT = (42, 120, 214)
_LINE_SAT = (27, 175, 122)
_LINE_UNKNOWN = (214, 57, 57)
_CHART_BG = (252, 252, 251)

# x축에서 라벨을 다는 주요 자릿수(로그 스케일 major tick). 나머지 촘촘한 eps는
# 이 사이에 라벨 없는 minor tick으로만 표시한다.
_LOG_MAJOR_EPS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]


def _chart_font(size: int):
    from PIL import ImageFont

    # 한글 라벨이 있어서 malgun.ttf(맑은 고딕, Windows 기본 탑재)를 우선 시도하고,
    # 없는 환경(리눅스 등)이면 arial -> Pillow 내장 비트맵 폰트 순으로 대체한다.
    for name in ("malgun.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _i_beam(draw, cx, y0, y1, color, cap_w=34, width=3):
    """세로선 + 위아래 가로 캡(I자 모양)의 floating range 마크 (범례용)."""
    draw.line([(cx, y0), (cx, y1)], fill=color, width=width)
    draw.line([(cx - cap_w / 2, y0), (cx + cap_w / 2, y0)], fill=color, width=width)
    draw.line([(cx - cap_w / 2, y1), (cx + cap_w / 2, y1)], fill=color, width=width)


def _split_beam(draw, cx, bottom_y, boundary_y, top_y, color_bottom, color_top, cap_w=34, width=3, top_cap_color=None):
    """세로선 하나가 경계에서 색만 바뀌는 형태. 캡은 맨 위/맨 아래에만 그려서
    간격이 있다 없다 하는 지저분함 없이 항상 이어 붙게 만든다.

    top_cap_color를 주면 맨 위 가로 캡만 그 색으로 그린다 (세로선 자체는 그대로
    color_top) -- 그 eps에 unknown 케이스가 있다는 걸 표시하는 용도."""
    draw.line([(cx, bottom_y), (cx, boundary_y)], fill=color_bottom, width=width)
    draw.line([(cx, boundary_y), (cx, top_y)], fill=color_top, width=width)
    draw.line([(cx - cap_w / 2, bottom_y), (cx + cap_w / 2, bottom_y)], fill=color_bottom, width=width)
    draw.line([(cx - cap_w / 2, top_y), (cx + cap_w / 2, top_y)], fill=top_cap_color or color_top, width=width)


def aggregate_by_eps(records: list[dict]) -> dict[float, dict[str, int]]:
    """상태는 sat/unsat/unknown 3가지뿐이므로 셋 다 따로 센다 (sat을
    total-unsat으로 유추하지 않는다 -- unknown이 섞이면 그 유추가 틀림)."""
    by_eps: dict[float, dict[str, int]] = {}
    for r in records:
        d = by_eps.setdefault(r["eps"], {"unsat": 0, "sat": 0, "unknown": 0, "total": 0})
        d["total"] += 1
        d[r["status"]] += 1
    return by_eps


def filter_major_eps(by_eps: dict[float, dict[str, int]]) -> dict[float, dict[str, int]]:
    """전체 eps 중 7개 주요 자릿수(1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1)만 골라낸다."""
    return {eps: d for eps, d in by_eps.items() if eps in _LOG_MAJOR_EPS}


def write_eps_summary_csv(by_eps: dict[float, dict[str, int]], out_path: pathlib.Path) -> None:
    """차트를 다시 그릴 때 46만+개 result 파일을 매번 다시 스캔하지 않도록,
    eps별 집계(sat/unsat/unknown 개수)만 따로 가벼운 CSV로 저장해둔다."""
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["eps", "total", "unsat", "sat", "unknown"])
        writer.writeheader()
        for eps in sorted(by_eps):
            d = by_eps[eps]
            writer.writerow({"eps": eps, "total": d["total"], "unsat": d["unsat"], "sat": d["sat"], "unknown": d["unknown"]})


def read_eps_summary_csv(path: pathlib.Path) -> dict[float, dict[str, int]]:
    by_eps: dict[float, dict[str, int]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            by_eps[float(row["eps"])] = {
                "total": int(row["total"]),
                "unsat": int(row["unsat"]),
                "sat": int(row["sat"]),
                "unknown": int(row["unknown"]),
            }
    return by_eps


def write_eps_ratio_chart(by_eps: dict[float, dict[str, int]], n_idx: int, out_path: pathlib.Path) -> None:
    """eps별 unsat/sat 비율(전체 idx 대비)을 floating range 차트(I-beam 스타일)로
    PNG 이미지로 저장한다 (Pillow만 사용, 새 의존성 없음). x 위치는 eps 값과 무관하게
    균등 간격(equal scale)으로 배치하고, 포인트마다 eps 라벨을 전부 단다 (로그
    스케일은 eps_unsat_line.png 쪽에서 담당)."""
    from PIL import Image, ImageDraw

    eps_sorted = sorted(by_eps)
    n = len(eps_sorted)
    if n == 0:
        return

    margin_l, margin_r, margin_t, margin_b = 60, 70, 90, 60
    width, height = 1600, 480
    chart_w = width - margin_l - margin_r
    chart_h = height - margin_t - margin_b
    baseline_y = margin_t + chart_h
    top_y = margin_t

    img = Image.new("RGB", (width, height), _CHART_BG)
    draw = ImageDraw.Draw(img)

    font_title = _chart_font(20)
    font_subtitle = _chart_font(13)
    font_small = _chart_font(9)

    draw.text((margin_l, 16), f"eps별 unsat / sat 비율 (전체 {n_idx}개 idx 대비)", fill=_CHART_INK, font=font_title)
    draw.text((margin_l, 42), "아래(파랑)=unsat 구간, 위(초록)=sat 구간", fill=_CHART_MUTED, font=font_subtitle)

    # legend
    lx, ly = margin_l, 64
    _i_beam(draw, lx + 10, ly, ly + 14, _LINE_UNSAT, cap_w=16, width=2)
    draw.text((lx + 26, ly + 1), "unsat", fill=_CHART_MUTED, font=font_small)
    lx2 = lx + 26 + draw.textlength("unsat", font=font_small) + 26
    _i_beam(draw, lx2 + 10, ly, ly + 14, _LINE_SAT, cap_w=16, width=2)
    draw.text((lx2 + 26, ly + 1), "sat", fill=_CHART_MUTED, font=font_small)
    lx3 = lx2 + 26 + draw.textlength("sat", font=font_small) + 26
    draw.line([(lx3, ly), (lx3 + 20, ly)], fill=_LINE_UNKNOWN, width=2)
    draw.text((lx3 + 26, ly + 1), "맨 위 캡이 빨강 = 그 eps에 unknown 존재", fill=_CHART_MUTED, font=font_small)

    xs = [margin_l + i * (chart_w / (n - 1)) for i in range(n)] if n > 1 else [margin_l + chart_w / 2]

    for x, eps in zip(xs, eps_sorted):
        d = by_eps[eps]
        unsat_ratio = d["unsat"] / d["total"] * 100 if d["total"] else 0.0
        sat_ratio = 100.0 - unsat_ratio
        boundary_y = baseline_y - (unsat_ratio / 100) * chart_h
        top_cap_color = _LINE_UNKNOWN if d.get("unknown", 0) > 0 else None

        _split_beam(draw, x, baseline_y, boundary_y, top_y, _LINE_UNSAT, _LINE_SAT, top_cap_color=top_cap_color)

        # 1e-06은 unsat 비율이 거의 100%라 소수점 1자리로 반올림하면 "100.0%/0.0%"로
        # 찍혀서 실제로 존재하는 소수의 sat 케이스가 안 보인다. 그 구간만 소수점
        # 2자리까지 보여줘서 0%가 아니라는 걸 드러낸다.
        decimals = 2 if eps == 1e-6 else 1
        u_label = f"{unsat_ratio:.{decimals}f}%"
        s_label = f"{sat_ratio:.{decimals}f}%"
        tw_u = draw.textlength(u_label, font=font_small)
        text_h = font_small.size + 2

        # 라벨은 경계(boundary) 높이에 두되, 선/캡과 겹치지 않도록 좌우로 살짝 띄운다.
        # (unsat=파랑은 왼쪽, sat=초록은 오른쪽). 캔버스 밖으로 나가지 않도록 clamp.
        label_cy = max(top_y + text_h / 2 + 4, min(baseline_y - text_h / 2 - 4, boundary_y))
        label_y = label_cy - text_h / 2
        side_gap = 6
        draw.text((x - side_gap - tw_u, label_y), u_label, fill=_LINE_UNSAT, font=font_small)
        draw.text((x + side_gap, label_y), s_label, fill=_LINE_SAT, font=font_small)

        eps_label = f"{eps:g}"
        tw2 = draw.textlength(eps_label, font=font_small)
        draw.text((x - tw2 / 2, baseline_y + 10), eps_label, fill=_CHART_MUTED, font=font_small)

    draw.line([(margin_l, baseline_y), (margin_l + chart_w, baseline_y)], fill=_CHART_BASELINE, width=1)

    draw.text((margin_l, height - 22), "x축: eps (L_inf, 균등 간격 배치)  ·  y축: 비율(%)", fill=_CHART_MUTED, font=font_small)

    img.save(out_path)


def write_eps_unsat_line_chart(by_eps: dict[float, dict[str, int]], n_idx: int, out_path: pathlib.Path) -> None:
    """eps별 unsat 비율을 로그 스케일 x축의 단일 선 그래프로 그린다.

    x 위치는 log10(eps)로 잡는다 (균등 간격이 아니라 실제 값 비율 반영). 자릿수가
    바뀌는 7개 지점(1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1)만 라벨 있는 굵은 눈금이고,
    그 사이 촘촘한 eps(0.0002~0.0009, 0.002~0.009)는 라벨 없는 짧은 눈금만 그린다."""
    import math

    from PIL import Image, ImageDraw

    eps_sorted = sorted(by_eps)
    if not eps_sorted:
        return

    margin_l, margin_r, margin_t, margin_b = 60, 30, 90, 60
    width, height = 900, 480
    chart_w = width - margin_l - margin_r
    chart_h = height - margin_t - margin_b
    baseline_y = margin_t + chart_h

    img = Image.new("RGB", (width, height), _CHART_BG)
    draw = ImageDraw.Draw(img)

    font_title = _chart_font(20)
    font_subtitle = _chart_font(13)
    font_small = _chart_font(9)

    draw.text((margin_l, 16), f"eps별 unsat 비율 (전체 {n_idx}개 idx 대비, 로그 스케일)", fill=_CHART_INK, font=font_title)
    draw.text((margin_l, 42), "x축: eps (log10 스케일)  ·  y축: unsat 비율(%)", fill=_CHART_MUTED, font=font_subtitle)

    log_min = math.log10(eps_sorted[0])
    log_max = math.log10(eps_sorted[-1])
    log_span = (log_max - log_min) or 1.0

    def x_for(eps: float) -> float:
        return margin_l + (math.log10(eps) - log_min) / log_span * chart_w

    # y축 그리드 (0/25/50/75/100%)
    for pct in (0, 25, 50, 75, 100):
        gy = baseline_y - pct / 100 * chart_h
        draw.line([(margin_l, gy), (margin_l + chart_w, gy)], fill=_CHART_GRID, width=1)
        label = f"{pct}%"
        tw = draw.textlength(label, font=font_small)
        draw.text((margin_l - tw - 8, gy - 6), label, fill=_CHART_MUTED, font=font_small)

    # x축 minor tick: 주요 자릿수가 아닌 나머지 촘촘한 eps
    major_set = set(_LOG_MAJOR_EPS)
    for eps in eps_sorted:
        if eps in major_set:
            continue
        x = x_for(eps)
        draw.line([(x, baseline_y), (x, baseline_y + 4)], fill=_CHART_MUTED, width=1)

    # x축 major tick + 라벨: 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1
    for eps in _LOG_MAJOR_EPS:
        if eps < eps_sorted[0] or eps > eps_sorted[-1]:
            continue
        x = x_for(eps)
        draw.line([(x, baseline_y), (x, baseline_y + 8)], fill=_CHART_INK, width=2)
        label = f"{eps:g}"
        tw = draw.textlength(label, font=font_small)
        draw.text((x - tw / 2, baseline_y + 12), label, fill=_CHART_INK, font=font_small)

    # 선 + 포인트 마커 (모든 eps, major/minor 구분 없이 하나로 이어 그림)
    points = []
    for eps in eps_sorted:
        d = by_eps[eps]
        ratio = d["unsat"] / d["total"] * 100 if d["total"] else 0.0
        points.append((x_for(eps), baseline_y - ratio / 100 * chart_h))

    draw.line(points, fill=_LINE_UNSAT, width=2, joint="curve")
    r = 3
    for x, y in points:
        draw.ellipse([x - r, y - r, x + r, y + r], fill=_LINE_UNSAT)

    draw.line([(margin_l, baseline_y), (margin_l + chart_w, baseline_y)], fill=_CHART_BASELINE, width=1)

    img.save(out_path)


def read_result(result_path: pathlib.Path) -> tuple[str, str]:
    """상태는 sat/unsat/unknown 3가지로만 취급한다. 아직 안 돌았거나(파일 없음/빈
    파일) NeuralSAT이 timeout/restart 등 sat/unsat 이외의 상태를 내도 전부
    unknown으로 묶는다."""
    if not result_path.exists() or result_path.stat().st_size == 0:
        return "unknown", ""
    first_line = result_path.read_text().splitlines()[0]
    parts = first_line.split(",", 1)
    status = parts[0] if parts[0] in ("sat", "unsat") else "unknown"
    runtime = parts[1] if len(parts) == 2 else ""
    return status, runtime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--summary-out", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--robustness-out", default=str(DEFAULT_ROBUSTNESS_SUMMARY))
    parser.add_argument("--chart-out", default=str(DEFAULT_CHART), help="eps별 unsat/sat 비율 range 차트(PNG) 출력 경로")
    parser.add_argument("--chart-7-out", default=str(DEFAULT_CHART_7), help="eps 7개(주요 자릿수)만 보여주는 range 차트(PNG) 출력 경로")
    parser.add_argument("--line-chart-out", default=str(DEFAULT_LINE_CHART), help="eps별 unsat 비율 로그스케일 선 그래프(PNG) 출력 경로")
    parser.add_argument("--eps-summary-out", default=str(DEFAULT_EPS_SUMMARY), help="eps별 집계(가벼운 CSV) 출력 경로")
    parser.add_argument(
        "--chart-only",
        action="store_true",
        help="result 파일들을 다시 스캔하지 않고, --eps-summary-out에 저장된 이전 집계로 차트만 빠르게 다시 그린다 "
        "(차트 디자인만 반복 조정할 때 사용).",
    )
    args = parser.parse_args()

    if args.chart_only:
        by_eps = read_eps_summary_csv(pathlib.Path(args.eps_summary_out))
        n_idx = max((d["total"] for d in by_eps.values()), default=0)
        write_eps_ratio_chart(by_eps, n_idx, pathlib.Path(args.chart_out))
        write_eps_ratio_chart(filter_major_eps(by_eps), n_idx, pathlib.Path(args.chart_7_out))
        write_eps_unsat_line_chart(by_eps, n_idx, pathlib.Path(args.line_chart_out))
        print(f"wrote {args.chart_out} (from cached {args.eps_summary_out}, no result rescan)")
        print(f"wrote {args.chart_7_out} (from cached {args.eps_summary_out}, no result rescan)")
        print(f"wrote {args.line_chart_out} (from cached {args.eps_summary_out}, no result rescan)")
        return

    with open(args.manifest, newline="") as f:
        rows = list(csv.DictReader(f))

    records = []
    for row in rows:
        status, runtime = read_result(pathlib.Path(row["result"]))
        records.append({"idx": int(row["idx"]), "eps": float(row["eps"]), "status": status, "runtime": runtime})

    with open(args.summary_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["idx", "eps", "status", "runtime"])
        writer.writeheader()
        for r in sorted(records, key=lambda r: (r["idx"], r["eps"])):
            writer.writerow(r)

    by_idx: dict[int, list[dict]] = {}
    for r in records:
        by_idx.setdefault(r["idx"], []).append(r)

    robustness_rows = []
    status_counts: dict[str, int] = {}
    for idx, recs in sorted(by_idx.items()):
        recs.sort(key=lambda r: r["eps"])
        robust_radius = None
        break_eps = None
        break_status = None
        anomaly = False
        broke = False
        for r in recs:
            status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
            if not broke:
                if r["status"] == "unsat":
                    robust_radius = r["eps"]
                else:
                    broke = True
                    break_eps = r["eps"]
                    break_status = r["status"]
            elif r["status"] == "unsat":
                anomaly = True

        robustness_rows.append(
            {
                "idx": idx,
                "robust_radius_eps": robust_radius if robust_radius is not None else "",
                "first_non_unsat_eps": break_eps if break_eps is not None else "",
                "first_non_unsat_status": break_status if break_status is not None else "",
                "anomaly_nonmonotonic": anomaly,
            }
        )

    with open(args.robustness_out, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["idx", "robust_radius_eps", "first_non_unsat_eps", "first_non_unsat_status", "anomaly_nonmonotonic"],
        )
        writer.writeheader()
        writer.writerows(robustness_rows)

    by_eps = aggregate_by_eps(records)
    write_eps_summary_csv(by_eps, pathlib.Path(args.eps_summary_out))
    write_eps_ratio_chart(by_eps, len(by_idx), pathlib.Path(args.chart_out))
    write_eps_ratio_chart(filter_major_eps(by_eps), len(by_idx), pathlib.Path(args.chart_7_out))
    write_eps_unsat_line_chart(by_eps, len(by_idx), pathlib.Path(args.line_chart_out))

    print(f"summary: {len(records)} instances across {len(by_idx)} data points")
    print(f"status counts: {status_counts}")
    print(f"wrote {args.summary_out}")
    print(f"wrote {args.robustness_out}")
    print(f"wrote {args.eps_summary_out}")
    print(f"wrote {args.chart_out}")
    print(f"wrote {args.chart_7_out}")
    print(f"wrote {args.line_chart_out}")


if __name__ == "__main__":
    main()
