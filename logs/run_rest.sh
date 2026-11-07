#!/bin/sh
# Remaining grid: Qwen 0.5B, Qwen 1.5B, Llama 1B, Llama 3B. Llama weights download in parallel.
cd "$(dirname "$0")/.."
export HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTHONUNBUFFERED=1
python -c "
from huggingface_hub import snapshot_download
import time
for r in ['meta-llama/Llama-3.2-1B-Instruct','meta-llama/Llama-3.2-3B-Instruct']:
    for k in range(10):
        try: snapshot_download(r, allow_patterns=['*.json','*.safetensors','*.txt','*.jinja']); print('OK',r,flush=True); break
        except Exception as e: print('retry',r,k,type(e).__name__,str(e)[:120],flush=True); time.sleep(20)
    else: print('GAVE UP',r,flush=True)
" > logs/download.log 2>&1 &
DL=$!
sh logs/run_model.sh Qwen/Qwen2.5-0.5B-Instruct > logs/qwen0.5b.log 2>&1
sh logs/run_model.sh Qwen/Qwen2.5-1.5B-Instruct > logs/qwen1.5b.log 2>&1
wait $DL
sh logs/run_model.sh meta-llama/Llama-3.2-1B-Instruct > logs/llama1b.log 2>&1
sh logs/run_model.sh meta-llama/Llama-3.2-3B-Instruct > logs/llama3b.log 2>&1
echo ALL DONE
