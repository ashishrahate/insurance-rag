# Closing a Colab GPU session cleanly

Companion to `docs/colab-gpu-plan.md` / `colab/colab_gpu_eval.ipynb`. Run this
checklist at the end of any Colab session using this repo. Nothing here
assumes Google Drive is mounted -- if it isn't, step 2 (downloading results
off the ephemeral VM) is the only step that actually matters; everything
else is good hygiene, not data-loss prevention.

## 1. Confirm the eval run actually finished

Don't shut anything down mid-run.

```bash
!ps aux | grep -E "python|uvicorn|ollama" | grep -v grep
```

If a `run_full_eval` (or any other) python process is still listed, wait for
it.

## 2. Download everything you need -- do this before touching any process

`logs/runs.jsonl` only exists once something has actually called
`log_run()` -- if the runtime was restarted since the last real
query/eval call and nothing hit the pipeline after that, the file won't
exist yet and the download will fail. Run one query first if needed:

```bash
!python -m src.ask "smoke damage claims" --show-chunks
!ls -la logs/
```

```python
from google.colab import files
files.download('logs/runs.jsonl')
files.download('data/eval/results.md')
files.download('data/eval/answer_eval_results.md')
files.download('judge.log')
files.download('ollama.log')
```

Also worth grabbing to skip re-scraping next time -- the exact corpus this
session used, zipped:

```bash
!zip -r ca_corpus.zip data/raw/ca data/processed/ca
```
```python
files.download('ca_corpus.zip')
```

## 3. Stop the background services cleanly

Frees GPU memory/VRAM before disconnect. Good practice even though the VM is
about to be destroyed anyway.

```bash
!pkill -f "uvicorn src.judge_service"
!pkill -f "ollama serve"
!sleep 2
!ps aux | grep -E "uvicorn|ollama" | grep -v grep   # should print nothing now
```

## 4. Confirm GPU memory released

```python
import torch
torch.cuda.empty_cache()
print(torch.cuda.memory_allocated() / 1e6, "MB allocated")
```
```bash
!nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

## 5. Save the notebook itself

File -> Download -> Download `.ipynb`, or File -> Save a copy in Drive if
you want it to persist without a manual download each time.

## 6. Fully release the T4

Runtime -> Disconnect and delete runtime -- not just closing the tab or
"Manage sessions". This matters on the free tier: disconnecting properly
returns the GPU allocation immediately rather than leaving it held until
Colab's idle timeout kicks in, which affects usage quota for next time.

## After closing

Bring the downloaded files back into the local repo (`logs/`,
`data/eval/*.md`) and fold the new rows in manually, or run:

```bash
python -m src.observability.report --file colab_runs.jsonl --group env
```

to compare the Colab run(s) against the local `local-cpu` baseline row for
row.
