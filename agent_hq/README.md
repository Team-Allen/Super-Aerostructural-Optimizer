# Agent HQ (scaffold)

This folder contains a minimal Local Agent HQ for running scripted tasks in the repo.

Files:
- `run_agent.py`: Simple Python runner that loads `tasks.json`, executes commands, and writes `status.json`.
- `tasks.json`: Example tasks (quick test + training dry-run).
- `run_agent.ps1`: PowerShell helper to run the agent on Windows.

Quick start (PowerShell):

```powershell
# Run all tasks sequentially
python agent_hq/run_agent.py --run-all

# Run only task id 1
python agent_hq/run_agent.py --task-id 1
```

How to add tasks:
- Edit `agent_hq/tasks.json` and add objects with keys: `id`, `name`, `cmd`, `workdir`.
- `workdir` defaults to repository root when empty.

Status and logs:
- Runner writes `agent_hq/status.json` with timestamps, return codes and captured stdout/stderr (trimmed).

Notes:
- The runner uses `subprocess.run(..., shell=True)` for convenience. Use careful commands and validate before adding to `tasks.json`.
- You can integrate `agent_hq` with CI/CD by calling the runner from GitHub Actions or similar.
