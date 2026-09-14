"""
Host PC Automated Task Runner for Volga IT 2026.
Watches D:\AIProjects\VolgaIT\.jobs for instructions dispatched by Antigravity from the laptop.
Executes tasks locally using the RTX 5080 and local CPU cores, piping output to log files.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
JOBS_DIR = BASE_DIR / ".jobs"
VENV_PYTHON = BASE_DIR / ".venv" / "Scripts" / "python.exe"

if not VENV_PYTHON.exists():
    VENV_PYTHON = Path(sys.executable)


def process_job(job_file: Path):
    try:
        with open(job_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[-] Error reading {job_file.name}: {e}")
        return

    if data.get("status") != "pending":
        return

    job_id = data.get("id", job_file.stem)
    cmd = data.get("command")
    cwd = data.get("cwd", str(BASE_DIR))
    log_file = JOBS_DIR / f"{job_id}.log"

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] >>> Starting Job: {job_id}")
    print(f"Command: {cmd}")

    data["status"] = "running"
    data["started_at"] = datetime.now().isoformat()
    with open(job_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # If command starts with python, substitute with local venv python
    if cmd.startswith("python "):
        cmd = f'"{VENV_PYTHON}" ' + cmd[7:]

    with open(log_file, "w", encoding="utf-8") as log_f:
        log_f.write(f"=== Job {job_id} started at {datetime.now().isoformat()} ===\n")
        log_f.write(f"Command: {cmd}\n")
        log_f.write(f"CWD: {cwd}\n")
        log_f.write(f"Python: {VENV_PYTHON}\n\n")
        log_f.flush()

        start_t = time.time()
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_f.write(line)
            log_f.flush()

        proc.wait()
        duration = time.time() - start_t
        log_f.write(f"\n=== Job {job_id} finished in {duration:.2f}s with code {proc.returncode} ===\n")

    data["status"] = "completed" if proc.returncode == 0 else "failed"
    data["returncode"] = proc.returncode
    data["duration_sec"] = duration
    data["completed_at"] = datetime.now().isoformat()

    with open(job_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] <<< Job {job_id} {data['status'].upper()} in {duration:.1f}s")


def main():
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 65)
    print("  Volga IT 2026 - Host Workstation Task Runner (RTX 5080)")
    print(f"  Root:   {BASE_DIR}")
    print(f"  Jobs:   {JOBS_DIR}")
    print(f"  Python: {VENV_PYTHON}")
    print("=" * 65)
    print("[*] Runner active and listening for incoming jobs...")

    while True:
        try:
            job_files = sorted(JOBS_DIR.glob("*.json"))
            for jf in job_files:
                try:
                    with open(jf, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    if meta.get("status") == "pending":
                        process_job(jf)
                except Exception:
                    pass
            time.sleep(1.0)
        except KeyboardInterrupt:
            print("\nHost Runner stopped by user.")
            break
        except Exception as e:
            print(f"Error in runner loop: {e}")
            time.sleep(2.0)


if __name__ == "__main__":
    main()
