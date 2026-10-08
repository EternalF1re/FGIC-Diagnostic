"""Generate final screen statistics after the background orchestrator finishes."""

from __future__ import annotations

import json
import subprocess
import sys
import time

from screen_core import HERE


def main() -> None:
    orchestrator = HERE / 'orchestrator_summary.json'
    watcher = HERE / 'completion_watcher_summary.json'
    while True:
        if orchestrator.exists():
            info = json.loads(orchestrator.read_text(encoding='utf-8'))
            if info.get('status') == 'ALL_COMPLETE':
                log_path = HERE / 'logs' / 'summarize_screen.log'
                with log_path.open('w', encoding='utf-8') as handle:
                    result = subprocess.run(
                        [sys.executable, str(HERE / 'summarize_screen.py')],
                        cwd=str(HERE.parent.parent), stdout=handle, stderr=subprocess.STDOUT,
                        check=False,
                    )
                status = 'SUMMARY_COMPLETE' if result.returncode == 0 else 'SUMMARY_FAILED'
                watcher.write_text(json.dumps({'status': status, 'exit_code': result.returncode}, indent=2), encoding='utf-8')
                raise SystemExit(result.returncode)
            if info.get('status') == 'FAILED_HALTED':
                watcher.write_text(json.dumps({'status': 'TRAINING_FAILED_HALTED'}, indent=2), encoding='utf-8')
                raise SystemExit(1)
        time.sleep(60)


if __name__ == '__main__':
    main()
