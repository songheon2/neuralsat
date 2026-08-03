"""
manifest.csv/run_batch.py 배치 파이프라인을 거치지 않고, 특정 (idx, eps) 하나만
바로 NeuralSAT으로 검증한다 (특정 인스턴스 재실행/디버깅용).

manifest.csv에 이미 그 (idx, eps) 행이 있으면 거기 기록된 net/data/k/abs_row를
그대로 재사용하고(같은 result 경로에 씀 -> summarize_results.py가 그대로 집계함),
없으면 generate_vnnlib.py와 동일한 기본값으로 새로 계산한다.

사용 예:
    python run_single.py --idx 29632 --eps 0.001
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import time

from data_split import test_row_range
from generate_vnnlib import DEFAULT_DATA, DEFAULT_NET, result_path_for
from run_batch import DEFAULT_MAIN_PY, DEFAULT_MANIFEST, get_session, make_vnnlib_file, run_one

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent


def find_in_manifest(manifest_path: pathlib.Path, idx: int, eps: float) -> dict | None:
    if not manifest_path.exists():
        return None
    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            if int(row["idx"]) == idx and float(row["eps"]) == eps:
                return row
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--idx", type=int, required=True, help="test-split 기준 상대 idx")
    parser.add_argument("--eps", type=float, required=True, help="L_inf eps")
    parser.add_argument("--net", default=None, help="Path to ONNX model (기본: manifest에 있으면 그 값, 없으면 generate_vnnlib.py 기본값)")
    parser.add_argument("--data", default=None, help="Path to pickle 데이터 (기본: 위와 동일 규칙)")
    parser.add_argument("--k", type=int, default=None, help="top-k (기본: 위와 동일 규칙, manifest에 없으면 8)")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="조회할 manifest.csv 경로")
    parser.add_argument("--main", default=str(DEFAULT_MAIN_PY), help="Path to neuralsat's src/main.py")
    parser.add_argument("--results-dir", default=str(SCRIPT_DIR / "results"), help="manifest에 없을 때 result/log 파일을 쓸 위치")
    parser.add_argument("--no-data-in-file", type=int, default=20000, help="Samples per original source file")
    parser.add_argument("--no-test-files", type=int, default=2, help="Number of trailing files reserved for test")
    parser.add_argument("--keep-vnnlib", action="store_true", help="생성된 vnnlib spec 파일을 실행 후 지우지 않고 남겨둔다 (디버깅용)")
    args = parser.parse_args()

    manifest_path = pathlib.Path(args.manifest)
    row = find_in_manifest(manifest_path, args.idx, args.eps)

    if row is not None:
        net = args.net or row["net"]
        data = args.data or row["data"]
        k = args.k if args.k is not None else int(row["k"])
        abs_row = int(row["abs_row"])
        result_path = pathlib.Path(row["result"])
        print(f"manifest에서 idx={args.idx} eps={args.eps} 발견 (net/data/k/abs_row 재사용)")
    else:
        net = args.net or str(DEFAULT_NET)
        data = args.data or str(DEFAULT_DATA)
        k = args.k if args.k is not None else 8
        test_start, _ = test_row_range(data, args.no_data_in_file, args.no_test_files)
        abs_row = test_start + args.idx
        result_path = result_path_for(pathlib.Path(args.results_dir), k, args.idx, args.eps)
        print(f"manifest에 없어서 새로 계산: net={net}, data={data}, k={k}, abs_row={abs_row}")

    log_path = result_path.with_suffix(".log")
    tmp_dir = SCRIPT_DIR / "vnnlib" / "_tmp_vnnlib"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    sess = get_session(net)
    spec_path = make_vnnlib_file(sess, data, abs_row, args.eps, k, tmp_dir)
    print(f"vnnlib spec: {spec_path}")

    start = time.time()
    try:
        returncode = run_one(pathlib.Path(args.main), net, spec_path, result_path, log_path, {})
    finally:
        if args.keep_vnnlib:
            print(f"vnnlib spec kept at {spec_path}")
        else:
            spec_path.unlink(missing_ok=True)
    elapsed = time.time() - start

    if returncode != 0 or not result_path.exists():
        print(f"[FAILED] returncode={returncode} ({elapsed:.1f}s) see {log_path}")
        return

    status_line = result_path.read_text().splitlines()[0] if result_path.stat().st_size > 0 else "<empty>"
    print(f"idx={args.idx} eps={args.eps} -> {status_line} ({elapsed:.1f}s)")
    print(f"result: {result_path}")
    print(f"log: {log_path}")


if __name__ == "__main__":
    main()
