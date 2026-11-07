# Internal-State Hallucination Detection in Small Language Models

This project investigates whether a model's hidden states can help detect factual errors beyond what can already be inferred from its answers and confidence. It evaluates four probe architectures on the same generations and compares them with progressively stronger output-based baselines.

The experiments cover five instruction-tuned Qwen 2.5 and Llama 3.2 models, ranging from 0.5B to 3B parameters. The main evaluation uses a fixed sample of 800 PopQA questions. Generated answers are labelled by normalized string matching against the reference answers.

## Approach

1. Prepare a fixed question set and generate answers with controlled decoding settings.
2. Cache the model's hidden states and train linear, MLP, layer-attention, and token-trajectory probes.
3. Compare each probe against a ladder of surface features, confidence features, and output-based benchmark detectors.
4. Measure the change in AUROC when the probe is added to each baseline, with bootstrap intervals and control checks.

The report describes an average gain of 0.126 AUROC over surface features, falling to 0.025 over the strongest benchmark level. Two of the 20 model–probe combinations had intervals excluding zero at that level. These are reported experimental results, not results reproduced by this repository upload.

## Repository layout

- `data/` — dataset loaders, raw files and prepared working sets.
- `src/` — extraction, probe models, training, baselines and evaluation.
- `scripts/` — data preparation, experiment runs, checks and final analysis.
- `Final_results/` — saved results and run status from the supplied project.
- `logs/` — logs and helper scripts from the original runs.
- `tests/` — checks for loaders, the baseline ladder and the analysis chain.

Activation caches and model weights are not included in the supplied archive. They need to be generated or downloaded before running the full analysis. Existing logs may contain paths from the original machine.

## Getting started

The code uses Python, PyTorch, Transformers, NumPy, pandas and scikit-learn. The dataset downloader also uses the Hugging Face `datasets` package. Install a suitable PyTorch build for your system and the remaining packages in a virtual environment. The archive does not include a pinned dependency file, so exact package versions from the original run are not specified.

```bash
pip install torch transformers numpy pandas scikit-learn datasets
python scripts/00_prepare_data.py --source popqa --n 800
python scripts/03_run_experiment.py --dataset popqa --model Qwen/Qwen2.5-0.5B-Instruct --arch layer_attn --device cuda
python scripts/06_baselines.py --dataset popqa --model Qwen/Qwen2.5-0.5B-Instruct --device cuda
python scripts/07_final_analysis.py --crossfit
```

Run commands from the repository root. Model downloads need network access; Llama checkpoints may require access approval and authentication. GPU memory, storage and runtime requirements depend on the checkpoint and cache settings. The prepared data included in the archive can be used with the existing working set; avoid rebuilding it when reusing existing activation caches, because the order must match. `scripts/09_verify_alignment.py` checks cache alignment.

For dataset download options, see `scripts/00_fetch_data.py`. Raw data defaults to `data/raw/`; set `HALLU_DATA` to use another location. The full multi-model workflow is in `scripts/run_review3.sh` and `scripts/run_review3.ps1`.

## Scope and limitations

The main results concern short factual answers on PopQA and models up to 3B parameters; they should not be assumed to extend to long-form answers or larger models. String matching can mark a correct paraphrase as wrong. The semantic entropy baseline uses normalized answer strings as an approximation rather than an entailment model. The report also notes the limited sample size and the exclusion of more expensive retrieval-based or larger-model judges.

## Project team

C Pranay Raj, G Shiva Sampath and Sudeep Reddy. Project supervised by Pallavi Reddy at Amrita Vishwa Vidyapeetham.
