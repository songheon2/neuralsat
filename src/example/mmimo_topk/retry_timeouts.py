"""
NeuralSAT을 기본 timeout(3600s)으로 돌렸다가 timeout/unknown으로 끝난 인스턴스들을,
더 긴 --timeout으로 재시도하기 위한 스크립트.

이미 만들어져 있는 .vnnlib 스펙 파일들이 모인 디렉토리(파일명 규칙:
mmimo_top{k}_idx{idx}_eps{eps}.vnnlib, generate_vnnlib.py/run_single.py와 동일)를
훑어서, 파일마다 NeuralSAT을 새 timeout으로 한 번씩 돌린다.

run_single.py/run_batch.py와 달리:
- vnnlib을 즉석 생성하지 않고 디렉토리에 이미 있는 파일을 그대로 사용한다.
- 실행 후에도 그 파일을 지우지 않는다 (사용자가 보관 중인 재시도 대상 목록이므로).
- --timeout을 명시적으로 받아서 main.py에 그대로 넘긴다 (기본값 3600s를 오버라이드
  하지 않는 run_batch.py/run_single.py와의 차이점이자 이 스크립트의 존재 이유).

결과는 기존 파이프라인과 동일하게 results/mmimo_top{k}_idx{idx}_eps{eps}.result
경로에 쓰므로(이미 있던 실패 결과는 main.py가 덮어씀), summarize_results.py를
다시 돌리면 자동으로 반영된다.

사용 예:
    python retry_timeouts.py --timeout 7200
    python retry_timeouts.py --timeout 10800 --workers 4
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import psutil

from generate_vnnlib import DEFAULT_NET

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_DIR = pathlib.Path(
    r"C:\AI_Verification\jnunnv\vnnlib\Baseline mMIMO FC H hard short 80 HTHNN_LAY2_491 RELU 20241018 PRUNED 0.93_NO_SIGMOID\fail_idx"
)
DEFAULT_MAIN_PY = SCRIPT_DIR.parents[1] / "main.py"  # .../neuralsat/src/main.py
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"


HEARTBEAT_INTERVAL = 3600  # 인스턴스당 이 주기(초)마다 "아직 실행 중" 로그를 찍는다
_POLL_INTERVAL = 15  # 프로세스 생존 여부를 얼마나 자주 확인할지 (초). 로그 주기와는 별개.


def format_duration(seconds: float) -> str:
    # run_batch.py와 마찬가지로 ASCII만 사용 (conda run의 cp949 콘솔에서 한글이 깨지는 문제 회피).
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def run_one_with_timeout(
    main_py: pathlib.Path, net: str, spec: pathlib.Path, result_path: pathlib.Path, log_path: pathlib.Path,
    timeout: float, extra_env: dict[str, str], tag: str,
) -> int:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(main_py),
        "--net", net,
        "--spec", str(spec),
        "--result_file", str(result_path),
        "--timeout", str(timeout),
        "--export_runtime",
        "--export_cex",
    ]
    env = os.environ.copy()
    env.update(extra_env)

    start = time.time()
    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)

        # subprocess.run처럼 그냥 기다리지 않고, 주기적으로 살아있는지 poll하면서
        # 1시간마다 "아직 실행 중"을 찍는다 (timeout을 몇 시간으로 늘려 돌리는
        # 스크립트라, 진행 중인지 죽었는지 화면만 봐서는 알 수 없어서 넣음).
        next_heartbeat = HEARTBEAT_INTERVAL
        while proc.poll() is None:
            time.sleep(_POLL_INTERVAL)
            elapsed = time.time() - start
            if elapsed >= next_heartbeat:
                print(f"[heartbeat] {tag} still running ({format_duration(elapsed)} elapsed, timeout={timeout:g}s)", flush=True)
                next_heartbeat += HEARTBEAT_INTERVAL

    return proc.returncode


def process_one(spec_path: str, main_py: str, net: str, results_dir: str, timeout: float, thread_env: dict[str, str]) -> tuple[str, str]:
    """워커 프로세스에서 인스턴스 하나를 처리한다. 결과 메시지를 (kind, text)로 반환한다."""
    spec_path_p = pathlib.Path(spec_path)
    result_path = pathlib.Path(results_dir) / (spec_path_p.stem + ".result")
    log_path = result_path.with_suffix(".log")
    tag = spec_path_p.stem

    print(f"[start] {tag} (timeout={timeout:g}s)", flush=True)
    start = time.time()
    returncode = run_one_with_timeout(pathlib.Path(main_py), net, spec_path_p, result_path, log_path, timeout, thread_env, tag)
    elapsed = time.time() - start

    if returncode != 0 or not result_path.exists():
        return "failed", f"[FAILED] {tag} returncode={returncode} ({elapsed:.1f}s) see {log_path}"

    status_line = result_path.read_text().splitlines()[0] if result_path.stat().st_size > 0 else "<empty>"
    return "ok", f"{tag} -> {status_line} ({elapsed:.1f}s)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default=str(DEFAULT_DIR), help="재시도할 .vnnlib 파일들이 있는 디렉토리")
    parser.add_argument("--timeout", type=float, default=7200, help="새로 적용할 timeout(초). 기존 기본값 3600s보다 크게 주는 용도 (기본 7200)")
    parser.add_argument("--net", default=str(DEFAULT_NET), help="Path to the ONNX model")
    parser.add_argument("--main", default=str(DEFAULT_MAIN_PY), help="Path to neuralsat's src/main.py")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR), help="result/log를 쓸 위치 (기본: 기존 파이프라인과 동일한 results/)")
    parser.add_argument(
        "--workers", type=int, default=None,
        help="동시에 몇 개 인스턴스를 병렬로 돌릴지. 기본: 물리 코어 수 - 1 (run_batch.py와 동일 기준). 1이면 순차 실행.",
    )
    parser.add_argument("--dry-run", action="store_true", help="실행 없이 대상 목록만 출력")
    args = parser.parse_args()

    spec_dir = pathlib.Path(args.dir)
    spec_files = sorted(spec_dir.glob("*.vnnlib"))
    if not spec_files:
        print(f"{spec_dir} 에 .vnnlib 파일이 없습니다")
        return

    physical_cores = psutil.cpu_count(logical=False) or os.cpu_count() or 2
    workers = args.workers if args.workers is not None else max(1, physical_cores - 1)

    print(f"{len(spec_files)}개 인스턴스, timeout={args.timeout:g}s, workers={workers}")

    if args.dry_run:
        for spec_path in spec_files:
            print(f"[dry-run] would run {spec_path.stem} (timeout={args.timeout:g}s)")
        return

    results_dir = pathlib.Path(args.results_dir)

    thread_env: dict[str, str] = {}
    if workers > 1:
        threads_per_worker = max(1, physical_cores // workers)
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            thread_env[var] = str(threads_per_worker)

    n_ok = n_failed = 0

    def consume(i: int, result: tuple[str, str]) -> None:
        nonlocal n_ok, n_failed
        kind, text = result
        if kind == "ok":
            n_ok += 1
        else:
            n_failed += 1
        print(f"[{i}/{len(spec_files)}] {text}")

    if workers <= 1:
        for i, spec_path in enumerate(spec_files, 1):
            consume(i, process_one(str(spec_path), args.main, args.net, str(results_dir), args.timeout, thread_env))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(process_one, str(spec_path), args.main, args.net, str(results_dir), args.timeout, thread_env): i
                for i, spec_path in enumerate(spec_files, 1)
            }
            for fut in as_completed(futures):
                consume(futures[fut], fut.result())

    print(f"done: {n_ok} completed, {n_failed} failed (timeout={args.timeout:g}s, workers={workers})")


if __name__ == "__main__":
    main()
