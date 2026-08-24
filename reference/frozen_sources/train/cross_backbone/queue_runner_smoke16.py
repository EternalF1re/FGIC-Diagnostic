"""Smoke worker wrapper requiring 16 technical training batches per stage."""
import subprocess
import queue_runner

_run = subprocess.run

def _run_with_smoke16(command, *args, **kwargs):
    command = list(command)
    if "train_one_entry.py" in " ".join(str(value) for value in command) and "smoke" in command:
        command.extend(["--smoke-train-batches", "16"])
    return _run(command, *args, **kwargs)

queue_runner.subprocess.run = _run_with_smoke16

if __name__ == "__main__":
    queue_runner.main()
