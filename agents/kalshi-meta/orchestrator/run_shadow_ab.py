#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON = sys.executable or "python3"


def _utc_day() -> str:
    return dt.datetime.now(dt.timezone.utc).date().isoformat()


def _clean_tag(tag: str) -> str:
    out = "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "-" for ch in str(tag or "").strip())
    return out.strip("-")


def _build_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = str(env.get("PYTHONPATH") or "").strip()
    entries: List[str] = [str(ROOT), str(REPO_ROOT)]
    if existing:
        entries.extend([p for p in existing.split(os.pathsep) if p])
    deduped: List[str] = []
    for p in entries:
        if p and p not in deduped:
            deduped.append(p)
    env["PYTHONPATH"] = os.pathsep.join(deduped)
    return env


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _start_process(cmd: List[str], *, log_path: Path, env: dict[str, str]) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = log_path.open("a", encoding="utf-8")
    fh.write(f"\n\n===== START {dt.datetime.now(dt.timezone.utc).isoformat()} =====\n")
    fh.flush()
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=fh,
        stderr=subprocess.STDOUT,
        text=True,
    )
    # attach handle for cleanup
    proc._chimera_log_handle = fh  # type: ignore[attr-defined]
    return proc


def _terminate_process(proc: Optional[subprocess.Popen], *, name: str) -> None:
    if proc is None:
        return
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=8.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5.0)
    except Exception as exc:
        print(f"[AB] terminate failed name={name} error={exc}")
    finally:
        fh = getattr(proc, "_chimera_log_handle", None)
        if fh is not None:
            try:
                fh.flush()
                fh.close()
            except Exception:
                pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Parallel Shadow A/B runner with shared Kalshi feeder.")
    parser.add_argument("--date", default="", help="Trading date token or ISO date.")
    parser.add_argument("--tag", default="", help="Run tag for output directory separation.")
    parser.add_argument("--poll-seconds", type=float, default=0.5, help="Shadow sniper poll seconds for both strategies.")
    parser.add_argument("--max-runtime-minutes", type=float, default=0.0, help="Optional runtime limit for each strategy (0 = continuous).")
    parser.add_argument("--size-contracts", type=int, default=1, help="Contracts per shadow order.")
    parser.add_argument("--all-categories", action="store_true", help="Enable full-slate discovery for both strategies.")
    parser.add_argument("--tickers", default="", help="Optional explicit ticker override for both strategies.")
    parser.add_argument("--control-config", default=str(ROOT / "config" / "profiles" / "control_odds_api.json"), help="Control config JSON path.")
    parser.add_argument("--experiment-config", default=str(ROOT / "config" / "profiles" / "experiment_boltodds.json"), help="Experiment config JSON path.")
    parser.add_argument("--feed-poll-seconds", type=float, default=0.5, help="Shared feeder loop interval.")
    parser.add_argument("--feed-ws-timeout-seconds", type=float, default=5.0, help="Shared feeder WS timeout.")
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "ab"), help="A/B output directory root.")
    parser.add_argument("--maker-mode", action="store_true", help="Explicitly run both child strategies in maker mode.")
    args = parser.parse_args()

    day = str(args.date or "").strip() or _utc_day()
    tag = _clean_tag(args.tag or "")
    run_id = f"{day}_{tag}" if tag else day

    out_root = Path(str(args.output_dir)).resolve() / run_id
    out_root.mkdir(parents=True, exist_ok=True)
    subprocess_env = _build_subprocess_env()
    maker_mode = bool(args.maker_mode or _env_flag("SHADOW_IS_MAKER", default=False))
    subprocess_env["SHADOW_IS_MAKER"] = ("1" if maker_mode else "0")
    print(f"[AB] PYTHONPATH={subprocess_env.get('PYTHONPATH', '')}")
    print(f"[AB] execution_mode={'maker' if maker_mode else 'default'}")

    control_state = out_root / "control_state.json"
    control_ledger = out_root / "control_shadow_ledger.csv"
    control_summary = out_root / "control_shadow_summary.md"
    control_hf_ledger = out_root / "control_hf_eval_ledger.csv"

    experiment_state = out_root / "experiment_state.json"
    experiment_ledger = out_root / "experiment_shadow_ledger.csv"
    experiment_summary = out_root / "experiment_shadow_summary.md"
    experiment_hf_ledger = out_root / "experiment_hf_eval_ledger.csv"

    feed_path = out_root / "shared_market_feed.json"

    feeder_cmd = [
        PYTHON,
        str(ROOT / "orchestrator" / "shared_kalshi_feeder.py"),
        "--state-path",
        str(control_state),
        "--state-path",
        str(experiment_state),
        "--feed-path",
        str(feed_path),
        "--poll-seconds",
        str(max(0.1, float(args.feed_poll_seconds))),
        "--ws-timeout-seconds",
        str(max(0.5, float(args.feed_ws_timeout_seconds))),
        "--max-runtime-minutes",
        str(max(0.0, float(args.max_runtime_minutes))),
    ]

    common = [
        "--date",
        day,
        "--poll-seconds",
        str(max(0.1, float(args.poll_seconds))),
        "--max-runtime-minutes",
        str(max(0.0, float(args.max_runtime_minutes))),
        "--size-contracts",
        str(max(1, int(args.size_contracts))),
        "--shared-feed-path",
        str(feed_path),
        "--shared-feed-max-age-seconds",
        str(max(0.5, float(args.feed_poll_seconds) * 6.0)),
        "--disable-hf-root-ledger",
    ]

    if bool(args.all_categories):
        common.append("--all-categories")
    if str(args.tickers or "").strip():
        common.extend(["--tickers", str(args.tickers).strip()])
    if maker_mode:
        common.extend(["--execution-mode", "maker"])

    control_cmd = [
        PYTHON,
        str(ROOT / "orchestrator" / "run_shadow.py"),
        "--config",
        str(Path(args.control_config).resolve()),
        "--tag",
        f"{run_id}_control",
        "--state-path",
        str(control_state),
        "--ledger-path",
        str(control_ledger),
        "--summary-path",
        str(control_summary),
        "--hf-root-ledger-path",
        str(control_hf_ledger),
        *common,
    ]

    experiment_cmd = [
        PYTHON,
        str(ROOT / "orchestrator" / "run_shadow.py"),
        "--config",
        str(Path(args.experiment_config).resolve()),
        "--tag",
        f"{run_id}_experiment",
        "--state-path",
        str(experiment_state),
        "--ledger-path",
        str(experiment_ledger),
        "--summary-path",
        str(experiment_summary),
        "--hf-root-ledger-path",
        str(experiment_hf_ledger),
        *common,
    ]

    print(f"[AB] run_id={run_id}")
    print(f"[AB] output_dir={out_root}")
    print(f"[AB] feeder_feed_path={feed_path}")

    max_restarts = 3
    restart_cooldown_s = 5.0
    role_specs: dict[str, dict[str, object]] = {
        "feeder": {
            "cmd": feeder_cmd,
            "log_path": out_root / "shared_feeder.log",
            "proc": None,
            "restarts": 0,
        },
        "control": {
            "cmd": control_cmd,
            "log_path": out_root / "control.log",
            "proc": None,
            "restarts": 0,
        },
        "experiment": {
            "cmd": experiment_cmd,
            "log_path": out_root / "experiment.log",
            "proc": None,
            "restarts": 0,
        },
    }

    try:
        role_specs["feeder"]["proc"] = _start_process(
            feeder_cmd,
            log_path=Path(role_specs["feeder"]["log_path"]),
            env=subprocess_env,
        )
        time.sleep(1.0)
        role_specs["control"]["proc"] = _start_process(
            control_cmd,
            log_path=Path(role_specs["control"]["log_path"]),
            env=subprocess_env,
        )
        time.sleep(1.0)
        role_specs["experiment"]["proc"] = _start_process(
            experiment_cmd,
            log_path=Path(role_specs["experiment"]["log_path"]),
            env=subprocess_env,
        )

        print(
            f"[AB] processes started feeder_pid={role_specs['feeder']['proc'].pid} "
            f"control_pid={role_specs['control']['proc'].pid} "
            f"experiment_pid={role_specs['experiment']['proc'].pid}"
        )

        while True:
            restarted = False
            for role, spec in role_specs.items():
                proc = spec.get("proc")
                if not isinstance(proc, subprocess.Popen):
                    continue
                rc = proc.poll()
                if rc is None:
                    continue
                restarts = int(spec.get("restarts") or 0)
                print(f"[AB] {role} exited rc={rc} restart_count={restarts}")
                _terminate_process(proc, name=role)
                if restarts >= max_restarts:
                    print(
                        f"[AB][CRITICAL] {role} exceeded max_restarts={max_restarts}; "
                        "shutting down fleet"
                    )
                    return int(rc or 1)
                print(
                    f"[AB] restarting role={role} cooldown_s={restart_cooldown_s:.1f} "
                    f"next_restart={restarts + 1}/{max_restarts}"
                )
                time.sleep(restart_cooldown_s)
                spec["restarts"] = restarts + 1
                spec["proc"] = _start_process(
                    spec["cmd"],
                    log_path=Path(spec["log_path"]),
                    env=subprocess_env,
                )
                restarted = True
                break
            if restarted:
                continue
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("[AB] interrupted")
        return 130
    finally:
        _terminate_process(role_specs["experiment"].get("proc"), name="experiment")
        _terminate_process(role_specs["control"].get("proc"), name="control")
        _terminate_process(role_specs["feeder"].get("proc"), name="feeder")


if __name__ == "__main__":
    raise SystemExit(main())
