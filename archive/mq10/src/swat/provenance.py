"""실행 출처(provenance) 기록: 시작 시점 HEAD·작업 트리 dirty 여부 (decisions.md P2d-R G2 결정 5번).

긴 실행은 시작할 때 start_meta() 를 한 번 부르고, 결과 meta 에 그대로 합친다(기존 run_commit 키는 유지:
'저장 시점 HEAD').
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RUN_START_DEF = "실행 시작 시점의 git HEAD (프로세스 시작 직후 기록)"
RUN_COMMIT_DEF = "결과 저장 시점의 git HEAD (실행 도중 커밋이 생기면 시작 시점과 다를 수 있음)"
DIRTY_DEF = "실행 시작 시점 src/ phases/ tests/ 에 커밋되지 않은 변경(추적 파일 수정·새 파일)이 있으면 true"


def _git(*args: str, cwd: Path = ROOT) -> str | None:
    try:
        r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def git_head(cwd: Path = ROOT) -> str | None:
    return _git("rev-parse", "HEAD", cwd=cwd)


def git_dirty(cwd: Path = ROOT, paths: tuple[str, ...] = ("src", "phases", "tests")) -> bool | None:
    out = _git("status", "--porcelain", "--", *paths, cwd=cwd)
    return None if out is None else bool(out)


def start_meta(cwd: Path = ROOT) -> dict:
    """시작 시점 출처. 결과 meta 에 **start_meta() 로 합친다."""
    return dict(run_start_commit=git_head(cwd), run_start_dirty=git_dirty(cwd),
                run_start_time=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                run_start_commit_def=RUN_START_DEF, run_start_dirty_def=DIRTY_DEF, run_commit_def=RUN_COMMIT_DEF)
