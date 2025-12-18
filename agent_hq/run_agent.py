"""
Agent HQ runner - simple task runner for local automation.

Usage:
  python agent_hq/run_agent.py --run-all
  python agent_hq/run_agent.py --task-id 1

This runner loads `tasks.json` (list of tasks), executes each task command,
captures stdout/stderr, and writes a `status.json` with timestamps and exit codes.

Designed for Windows PowerShell default shell but also works on other shells.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
HQ_DIR = Path(__file__).resolve().parent
TASK_FILE = HQ_DIR / "tasks.json"
STATUS_FILE = HQ_DIR / "status.json"


def load_tasks():
    if not TASK_FILE.exists():
        print(f"No tasks.json found at {TASK_FILE}")
        return []
    with open(TASK_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def write_status(status):
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2, default=str)


def run_task(task):
    task_id = task.get("id")
    name = task.get("name")
    cmd = task.get("cmd")
    workdir = task.get("workdir") or str(ROOT)

    print(f"\n=== Running task {task_id}: {name} ===")
    print(f"Workdir: {workdir}")
    print(f"Cmd: {cmd}\n")

    started = datetime.utcnow().isoformat() + "Z"
    try:
        # Use shell=True so commands like `python training/train_gnn.py` work.
        # On Windows this will use cmd/powershell depending on environment.
        proc = subprocess.run(cmd, shell=True, cwd=workdir, capture_output=True, text=True)
        ret = {
            "id": task_id,
            "name": name,
            "cmd": cmd,
            "workdir": workdir,
            "start_time": started,
            "end_time": datetime.utcnow().isoformat() + "Z",
            "returncode": proc.returncode,
            "stdout": proc.stdout[:10000],
            "stderr": proc.stderr[:10000],
        }
        print(proc.stdout)
        if proc.returncode != 0:
            print(f"Task {task_id} exited with code {proc.returncode}")
    except Exception as e:
        ret = {
            "id": task_id,
            "name": name,
            "cmd": cmd,
            "workdir": workdir,
            "start_time": started,
            "end_time": datetime.utcnow().isoformat() + "Z",
            "returncode": -1,
            "stdout": "",
            "stderr": str(e),
        }
        print(f"Exception while running task {task_id}: {e}")

    return ret


def main():
    parser = argparse.ArgumentParser(description="Agent HQ runner")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-all", action="store_true", help="Run all tasks in tasks.json")
    group.add_argument("--task-id", type=int, help="Run a single task by id")
    args = parser.parse_args()

    tasks = load_tasks()
    if not tasks:
        print("No tasks found. Create agent_hq/tasks.json first.")
        sys.exit(1)

    status = {"run_time": datetime.utcnow().isoformat() + "Z", "tasks": []}

    if args.run_all:
        for task in tasks:
            result = run_task(task)
            status["tasks"].append(result)
            write_status(status)
    else:
        task = next((t for t in tasks if t.get("id") == args.task_id), None)
        if not task:
            print(f"Task id {args.task_id} not found in tasks.json")
            sys.exit(2)
        result = run_task(task)
        status["tasks"].append(result)
        write_status(status)

    print("\nAll requested tasks completed. Status written to agent_hq/status.json")


if __name__ == "__main__":
    main()
