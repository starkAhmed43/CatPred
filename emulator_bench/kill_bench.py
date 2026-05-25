import argparse
import os
import signal
import subprocess
import time
from pathlib import Path


DEFAULT_PATTERNS = [
    "emulator_bench/precompute_features.py",
    "emulator_bench/launch_parallel_retrain.py",
    "emulator_bench/train_single_target_tvt.py",
    "emulator_bench/cache_embeddings.py",
    "emulator_bench/align_structures.py",
]
REPO_ROOT = Path(__file__).resolve().parents[1]


def _user_processes() -> list[dict]:
    output = subprocess.check_output(
        ["ps", "-u", str(os.getuid()), "-o", "pid=,pgid=,command="],
        text=True,
    )
    processes = []
    for line in output.splitlines():
        parts = line.strip().split(maxsplit=2)
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            pgid = int(parts[1])
        except ValueError:
            continue
        processes.append({"pid": pid, "pgid": pgid, "command": parts[2]})
    return processes


def _process_cwd(pid: int) -> Path | None:
    try:
        return Path(os.readlink(f"/proc/{pid}/cwd")).resolve()
    except Exception:
        return None


def _is_under(path: Path | None, root: Path) -> bool:
    if path is None:
        return False
    return path == root or root in path.parents


def _matches(patterns: list[str], repo_root: Path, all_repos: bool) -> list[dict]:
    own_pid = os.getpid()
    out = []
    for process in _user_processes():
        command = process["command"]
        if process["pid"] == own_pid or "emulator_bench/kill_bench.py" in command:
            continue
        cwd = _process_cwd(process["pid"])
        if not all_repos and not _is_under(cwd, repo_root):
            continue
        if any(pattern in command for pattern in patterns):
            process["cwd"] = str(cwd) if cwd is not None else "unknown"
            out.append(process)
    return out


def _is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _send(processes: list[dict], sig: signal.Signals, process_groups: bool) -> None:
    seen_groups = set()
    for process in processes:
        try:
            if process_groups:
                pgid = int(process["pgid"])
                if pgid in seen_groups:
                    continue
                seen_groups.add(pgid)
                os.killpg(pgid, sig)
            else:
                os.kill(int(process["pid"]), sig)
        except ProcessLookupError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="List or terminate CatPred emulator_bench processes for this user.")
    parser.add_argument("--yes", action="store_true", help="Actually terminate matching processes. Without this, only lists.")
    parser.add_argument("--patterns", nargs="+", default=DEFAULT_PATTERNS)
    parser.add_argument("--repo_root", default=str(REPO_ROOT), help="Only match processes running under this repo root.")
    parser.add_argument("--all_repos", action="store_true", help="Do not filter by process working directory.")
    parser.add_argument("--process_groups", action="store_true", help="Kill process groups instead of individual PIDs.")
    parser.add_argument("--grace_seconds", default=10.0, type=float)
    parser.add_argument("--no_kill", action="store_true", help="Do not escalate to SIGKILL after grace period.")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).expanduser().resolve()
    processes = _matches(args.patterns, repo_root, args.all_repos)
    if not processes:
        scope = "all repos" if args.all_repos else str(repo_root)
        print(f"[kill_bench] no matching processes in scope: {scope}")
        return

    print("[kill_bench] matching processes:")
    for process in processes:
        print(f"  pid={process['pid']} pgid={process['pgid']} cwd={process['cwd']} {process['command']}")

    if not args.yes:
        print("\n[kill_bench] dry run only. Re-run with --yes to terminate these PIDs.")
        return

    _send(processes, signal.SIGTERM, args.process_groups)
    deadline = time.monotonic() + max(0.0, args.grace_seconds)
    while time.monotonic() < deadline:
        if not any(_is_alive(process["pid"]) for process in processes):
            print("[kill_bench] terminated")
            return
        time.sleep(0.25)

    survivors = [process for process in processes if _is_alive(process["pid"])]
    if not survivors:
        print("[kill_bench] terminated")
        return

    if args.no_kill:
        print("[kill_bench] still alive after SIGTERM:")
        for process in survivors:
            print(f"  pid={process['pid']} pgid={process['pgid']} {process['command']}")
        return

    print("[kill_bench] escalating to SIGKILL:")
    for process in survivors:
        print(f"  pid={process['pid']} pgid={process['pgid']} {process['command']}")
    _send(survivors, signal.SIGKILL, args.process_groups)


if __name__ == "__main__":
    main()
