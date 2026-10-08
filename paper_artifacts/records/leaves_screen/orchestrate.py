"""Run the 15 permitted #0/#1/#2 fold jobs over two local GPUs."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time

from screen_core import HERE, VARIANTS


def tasks():
    return [(variant, fold) for fold in range(5) for variant in VARIANTS]


def manifest(variant: str, fold: int) -> dict:
    path = HERE / variant / f'fold_{fold}' / 'run_manifest.json'
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return {'status': 'UNREADABLE'}


def write_ledger(running: dict, queue: list, halted: bool) -> None:
    rows = []
    queued = set(queue)
    for variant, fold in tasks():
        info = manifest(variant, fold)
        key = (variant, fold)
        status = info.get('status', 'PENDING')
        device = ''
        pid = ''
        if key in running:
            status = 'RUNNING'
            device = running[key]['device']
            pid = running[key]['process'].pid
        elif key in queued:
            status = 'QUEUED' if not halted else 'HALTED_AFTER_FAILURE'
        rows.append({
            'variant': variant, 'fold': fold, 'status': status, 'device': device,
            'pid': pid, 'split_id': info.get('split_id', f'skf42_fold{fold}'),
            'split_random_state': info.get('split_random_state', 42),
            'training_seed': info.get('training_seed', 42),
            'best_stage1_epoch_zero_based': info.get('best_stage1_epoch_zero_based', ''),
            'best_stage1_accuracy': info.get('best_stage1_accuracy', ''),
            'best_stage2_epoch_zero_based': info.get('best_stage2_epoch_zero_based', ''),
            'best_stage2_accuracy': info.get('best_stage2_accuracy', ''),
            'elapsed_seconds': info.get('elapsed_seconds', ''),
            'error': info.get('error', ''),
        })
    with (HERE / 'run_ledger.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--devices', nargs='+', default=['cuda:0', 'cuda:1'])
    args = parser.parse_args()
    audit_path = HERE / 'architecture_audit' / 'audit_summary.json'
    if not audit_path.exists() or not json.loads(audit_path.read_text(encoding='utf-8')).get('all_passed'):
        raise RuntimeError('Architecture audit is absent or failed; refusing formal training')
    queue = [(v, f) for v, f in tasks() if manifest(v, f).get('status') != 'COMPLETE']
    running = {}
    halted = False
    logs = HERE / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    write_ledger(running, queue, halted)
    while queue or running:
        used_devices = {job['device'] for job in running.values()}
        for device in args.devices:
            if halted or not queue or device in used_devices:
                continue
            variant, fold = queue.pop(0)
            log_path = logs / f'{variant}_fold_{fold}.log'
            handle = log_path.open('w', encoding='utf-8', buffering=1)
            command = [sys.executable, str(HERE / 'train_one.py'), '--variant', variant, '--fold', str(fold), '--device', device]
            process = subprocess.Popen(command, cwd=str(HERE.parent.parent), stdout=handle, stderr=subprocess.STDOUT)
            running[(variant, fold)] = {'process': process, 'device': device, 'handle': handle, 'log': str(log_path)}
            used_devices.add(device)
            print(json.dumps({'event': 'START', 'variant': variant, 'fold': fold, 'device': device, 'pid': process.pid}), flush=True)
        time.sleep(10)
        for key, job in list(running.items()):
            code = job['process'].poll()
            if code is None:
                continue
            job['handle'].close()
            del running[key]
            print(json.dumps({'event': 'EXIT', 'variant': key[0], 'fold': key[1], 'device': job['device'], 'exit_code': code, 'log': job['log']}), flush=True)
            if code != 0:
                halted = True
        write_ledger(running, queue, halted)
        if halted and not running:
            break
    summary = {'status': 'FAILED_HALTED' if halted else 'ALL_COMPLETE', 'allowed_tasks': 15, 'variants': list(VARIANTS)}
    (HERE / 'orchestrator_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if halted:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
