# Dynamic GPU scheduling

Independent fold/seed jobs can share one task queue instead of being statically divided between GPUs. Each idle GPU claims the next pending task, so faster devices naturally complete more work without changing seeds, splits, configs or output paths.

## Rules

1. Freeze every task ID, command, seed, fold, config and output directory before launch. Only the claiming device may vary.
2. Add `estimated_seconds` when historical timings exist; the scheduler dispatches longer tasks first to reduce tail latency.
3. Run at most one formal training task per GPU.
4. After any non-zero task exit, stop dispatching new work while allowing already-running tasks to finish.
5. Use per-task logs, an append-only event stream and an atomic ledger. Existing formal outputs are never overwritten.
6. Run each experiment's preflight audit before starting the queue.

## Task file

Commands may contain `{python}`, `{device}` and `{task_id}` placeholders.

```json
{
  "tasks": [
    {
      "task_id": "model_a_fold_0",
      "command": ["{python}", "-m", "fgic_diagnostic.cli", "train", "--fold", "0", "--device", "{device}"],
      "estimated_seconds": 14400,
      "metadata": {"fold": 0, "variant": "model_a"}
    }
  ]
}
```

## Launch

```powershell
python scripts\dynamic_gpu_queue.py `
  --tasks-json path\to\tasks.json `
  --output-dir outputs\scheduler `
  --work-dir . `
  --devices cuda:0 cuda:1
```

Always use a new scheduler output directory. Resumption must be explicit: audit completed manifests first, then pass only verified completed task IDs. A partial directory is never treated as a completed result.

