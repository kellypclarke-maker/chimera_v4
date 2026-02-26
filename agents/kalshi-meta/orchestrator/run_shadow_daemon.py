#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable or "python"


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _utc_day() -> str:
    return _utc_now().date().isoformat()


def _day_for_timezone(tz_name: str) -> str:
    name = str(tz_name or "UTC").strip() or "UTC"
    try:
        tz = ZoneInfo(name)
    except Exception:
        tz = dt.timezone.utc
    return dt.datetime.now(tz).date().isoformat()


def _clean_tag(tag: str) -> str:
    t = re.sub(r"[^A-Za-z0-9_-]+", "-", str(tag or "").strip()).strip("-")
    return t


def _run_cmd(cmd: List[str], *, timeout_seconds: float) -> Tuple[bool, int, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            timeout=max(1.0, float(timeout_seconds)),
        )
    except subprocess.TimeoutExpired as exc:
        return (False, 124, str(exc.stdout or ""), f"timeout_after_s={float(timeout_seconds):.1f}")
    ok = proc.returncode == 0
    return (ok, proc.returncode, proc.stdout or "", proc.stderr or "")


def _parse_selected_candidates(stdout: str) -> int:
    m = re.search(r"selected_candidates=(\d+)", str(stdout or ""))
    if not m:
        return 0
    try:
        return int(m.group(1))
    except Exception:
        return 0


def _read_shadow_counts(day_key: str) -> Dict[str, int]:
    out = {"open": 0, "filled": 0, "resolved": 0}
    path = ROOT / "reports" / "shadow" / f"{day_key}_shadow_ledger.csv"
    if not path.exists():
        return out
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                status = str(row.get("status") or "").strip().lower()
                if status in out:
                    out[status] += 1
    except Exception:
        return out
    return out


def _write_status(payload: Dict[str, Any], *, tag: str) -> Path:
    suffix = f"_{tag}" if str(tag or "").strip() else ""
    out_path = ROOT / "reports" / "ops" / f"status{suffix}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def _acquire_lock(lock_path: Path) -> int:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    payload = {
        "pid": os.getpid(),
        "started_ts": _utc_now().isoformat(),
    }
    os.write(fd, json.dumps(payload).encode("utf-8"))
    os.close(fd)
    return fd


def _run_cycle(args: argparse.Namespace) -> Dict[str, Any]:
    tag = _clean_tag(str(getattr(args, "tag", "") or ""))
    day = _day_for_timezone(str(getattr(args, "timezone", "UTC") or "UTC"))
    day_key = f"{day}_{tag}" if tag else day
    cycle_ts = _utc_now().isoformat()

    commands: List[Dict[str, Any]] = []
    ok = True
    error_message = ""

    run_daily_cmd = [
        PY,
        str(ROOT / "orchestrator" / "run_daily.py"),
        "--config",
        str(args.config),
        "--date",
        day,
    ]
    if tag:
        run_daily_cmd.extend(["--tag", tag])
    c_ok, rc, out, err = _run_cmd(run_daily_cmd, timeout_seconds=float(args.command_timeout_seconds))
    run_daily_rec = {"name": "run_daily", "cmd": run_daily_cmd, "ok": c_ok, "return_code": rc}
    commands.append(run_daily_rec)
    if not c_ok:
        existing_candidates = ROOT / "reports" / "daily" / f"{day_key}_candidates.csv"
        if existing_candidates.exists():
            run_daily_rec["ok"] = True
            run_daily_rec["fallback"] = "used_existing_candidates"
            run_daily_rec["fallback_candidates_csv"] = str(existing_candidates)
            run_daily_rec["run_daily_return_code"] = rc
        else:
            ok = False
            error_message = f"run_daily failed: rc={rc}"

    queue_size = 0
    if ok:
        queue_cmd = [
            PY,
            str(ROOT / "orchestrator" / "build_research_queue.py"),
            "--config",
            str(args.config),
            "--date",
            day,
            "--top-n",
            str(int(args.queue_top_n)),
            "--max-close-hours",
            str(float(args.queue_max_close_hours)),
            "--min-ev-dollars",
            str(float(args.queue_min_ev_dollars)),
        ]
        if tag:
            queue_cmd.extend(["--tag", tag])
        if bool(args.include_already_closed):
            queue_cmd.append("--include-already-closed")
        c_ok, rc, out, err = _run_cmd(queue_cmd, timeout_seconds=float(args.command_timeout_seconds))
        queue_size = _parse_selected_candidates(out)
        commands.append({"name": "build_research_queue", "cmd": queue_cmd, "ok": c_ok, "return_code": rc, "selected_candidates": queue_size})
        if not c_ok:
            ok = False
            error_message = f"build_research_queue failed: rc={rc}"

    if ok:
        shadow_cmd = [
            PY,
            str(ROOT / "orchestrator" / "run_shadow.py"),
            "--config",
            str(args.config),
            "--date",
            day,
            "--poll-seconds",
            str(int(args.shadow_poll_seconds)),
            "--max-runtime-minutes",
            str(float(args.shadow_max_runtime_minutes)),
        ]
        if tag:
            shadow_cmd.extend(["--tag", tag])
        c_ok, rc, out, err = _run_cmd(shadow_cmd, timeout_seconds=float(args.command_timeout_seconds))
        commands.append({"name": "run_shadow", "cmd": shadow_cmd, "ok": c_ok, "return_code": rc})
        if not c_ok:
            ok = False
            error_message = f"run_shadow failed: rc={rc}"

    if ok:
        rollup_cmd = [PY, str(ROOT / "orchestrator" / "rollup_shadow.py")]
        c_ok, rc, out, err = _run_cmd(rollup_cmd, timeout_seconds=float(args.command_timeout_seconds))
        commands.append({"name": "rollup_shadow", "cmd": rollup_cmd, "ok": c_ok, "return_code": rc})
        if not c_ok:
            ok = False
            error_message = f"rollup_shadow failed: rc={rc}"

    shadow_counts = _read_shadow_counts(day_key)

    status = {
        "timestamp": cycle_ts,
        "day": day,
        "day_key": day_key,
        "timezone": str(args.timezone),
        "tag": tag,
        "config_path": str(args.config),
        "last_run_status": "ok" if ok else "fail",
        "last_error": error_message,
        "queue": {
            "size": int(queue_size),
            "max_close_hours": float(args.queue_max_close_hours),
            "top_n": int(args.queue_top_n),
            "include_already_closed": bool(args.include_already_closed),
        },
        "shadow_counts": shadow_counts,
        "commands": commands,
    }
    status_path = _write_status(status, tag=tag)
    status["status_path"] = str(status_path)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description="Unattended shadow daemon for daily ingest + queue + weather shadow + rollup.")
    parser.add_argument("--config", default=str(ROOT / "config" / "defaults.json"), help="Path to config JSON.")
    parser.add_argument("--tag", default="", help="Optional tag suffix for outputs and lock/status files.")
    parser.add_argument("--timezone", default="UTC", help="Date timezone (default: UTC).")
    parser.add_argument("--interval-minutes", type=float, default=15.0, help="Loop interval in minutes (ignored with --once).")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit.")
    parser.add_argument("--queue-top-n", type=int, default=20, help="Queue size per cycle.")
    parser.add_argument("--queue-max-close-hours", type=float, default=72.0, help="Queue close horizon in hours.")
    parser.add_argument("--queue-min-ev-dollars", type=float, default=0.01, help="Queue min EV dollars.")
    parser.add_argument("--include-already-closed", action="store_true", help="Queue postmortem mode.")
    parser.add_argument("--command-timeout-seconds", type=float, default=900.0, help="Per-subcommand timeout (default: 900).")
    parser.add_argument("--shadow-poll-seconds", type=int, default=30, help="Shadow weather poll interval.")
    parser.add_argument("--shadow-max-runtime-minutes", type=float, default=1.0, help="Shadow weather burst runtime.")
    args = parser.parse_args()
    args.config = str(Path(str(args.config)).resolve())
    args.tag = _clean_tag(str(args.tag or ""))

    suffix = f"_{args.tag}" if str(args.tag) else ""
    lock_path = ROOT / "data" / "ops" / f"daemon{suffix}.lock"
    try:
        _acquire_lock(lock_path)
    except FileExistsError:
        print(f"daemon lock exists: {lock_path}")
        return 1

    try:
        while True:
            status = _run_cycle(args)
            print(json.dumps(status, indent=2))
            if args.once:
                break
            sleep_seconds = max(1.0, float(args.interval_minutes) * 60.0)
            time.sleep(sleep_seconds)
        return 0
    finally:
        try:
            if lock_path.exists():
                lock_path.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())

