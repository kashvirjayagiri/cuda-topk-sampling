# cuda-topk-sampling

EE 451: compare custom CUDA full bitonic sort and hierarchical Top-k, feeding a common CUDA token sampler. Compare correctness with an explicitly tie-broken reference and performance with PyTorch Top-k. Model inference is input preparation, never part of kernel timing.

## Dataset instructions

**The real-model dataset is already collected and shared through GitHub. Teammates should clone/pull and load the saved tensors; no model download or regeneration is needed.** Start with [the CARC teammate quickstart](data/README.md#teammates-start-here-on-carc). Collection commands below are for optional regeneration.

See [data/README.md](data/README.md) for dataset contents, CARC environment setup, loading examples, regeneration, and GitHub sharing instructions.

## First dataset: ten groups of real DistilGPT2 logits

`data/prompts/groups.json` contains 10 categories with 32 unique prompts each (320 total): everyday, science, math, code, history, story, dialogue, facts, procedures, and engineering. These are controlled starter prompts, not a representative language benchmark or guaranteed probability distributions. Category labels describe prompt subjects; they do not guarantee peaked/uniform outputs.

Each prompt supplies one raw next-token vector over all **50,257 vocabulary entries**. Each group becomes a contiguous FP32 tensor `[32, 50257]`, approximately 6.4 MB; all ten groups total approximately 64.3 MB of raw tensor storage. No training, softmax, temperature, or Top-k filtering is applied during collection.

## CARC setup and collection

Run inside an allocated GPU job/session, using your team's existing CUDA-enabled PyTorch environment. Use the CARC partition, account, and module settings assigned to your team; this repo does not assume those values.

```bash
nvidia-smi
nvcc --version
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python scripts/collect_logits.py
```

The first collection needs network access to download model/tokenizer files from Hugging Face. If GPU nodes cannot download, populate the Hugging Face cache on a network-enabled node first and make that cache available to the job. Model downloads are larger than the collected dataset.

Smoke-test just one group before collecting everything:

```bash
python scripts/collect_logits.py --groups 01_everyday --output data/logits/smoke
```

Default device is CUDA. CPU collection is supported for development with `--device cpu`; it is not GPU benchmarking. Default output is `data/logits/distilgpt2/`. Existing outputs are never overwritten: choose a new output directory for another collection. An interrupted collection may leave completed group files; rerun into a new directory.

The collector resolves the model commit and uses it for the tokenizer too. `manifest.json` records versions, device, commit, prompt-file hash, tensor shapes, and file hashes. For another collection using the same model revision:

```bash
python scripts/collect_logits.py --revision COMMIT_FROM_MANIFEST --output data/logits/repeated
```

Record the installed environment too:

```bash
python -m pip freeze > data/logits/distilgpt2/environment.txt
```

Everyone benchmarks the same saved tensors. The team shares the fixed dataset through GitHub; generated outputs are ignored by default, so adding the dataset requires explicitly forcing the ten files and manifest into Git. See [dataset sharing instructions](data/README.md#share-through-github). Do not commit model downloads or smoke-test outputs.

## Load inputs

```python
import torch

payload = torch.load(
    'data/logits/distilgpt2/01_everyday.pt',
    map_location='cpu', weights_only=True,
)
logits = payload['logits']  # [32, 50257], FP32
B, k = 8, 10
scores = logits[:B].contiguous().to('cuda')
g = torch.Generator(device='cuda').manual_seed(451)
u = torch.rand(B, generator=g, device='cuda')
# Pass the same scores and u to every method. Transfer/setup is outside timing.
```

Use B = 1, 8, 32 and k = 1, 10, 50, 100. Use all 32 rows for correctness; for B = 1 or 8, rotate through multiple row windows rather than treating one prompt as the whole category. Benchmark all groups and report variation. These categories are workloads, not ten statistical repetitions; repeat timed runs separately.

For the full-sort baseline, pad internally to 65,536 with negative infinity and invalid IDs. Report real input V = 50,257. Padding must never reach outputs. Keep original vocabulary IDs.

## Parallel implementation contract

- Selectors: contiguous GPU FP32 `[B,V]` plus k; return GPU FP32 values and int32 original IDs `[B,k]`.
- Sort descending by score, then ascending token ID. Exactly k outputs, input unchanged, supplied CUDA stream, working buffers preallocated.
- Sampler: selected values/IDs, positive temperature, and supplied uniform values `[B]` in `[0,1)`; return one original ID per row.
- Sampler uses stable softmax, parallel prefix-sum, and first CDF entry exceeding u, with rounding-safe final selection.
- Reference explicitly breaks ties; selected IDs must match exactly, values use 1e-5 absolute and relative tolerance.
- Initial input scope: finite FP32 scores. Test ties, partial tiles, padding, k=1, non-power-of-two k, and multiple batches.
- CUDA-event timings after warm-up; exclude model inference, allocation, CPU transfers, and correctness checks. Report selection and end-to-end latency using identical inputs and the same sampler.

## Ownership and milestones

- Kashvi: full GPU bitonic sort, build setup, integration.
- John: hierarchical Top-k, merge stages, limited tuning and memory analysis.
- Joel: shared CUDA sampler, Python reference/correctness harness, benchmark runner.
- Everyone: component tests, assigned experiments, plots/analysis, own report section and presentation material. Rebalance on November 9 based on actual effort.

November 6: tiny working components and agreed interfaces. November 9: independently working components and first integration. November 11: integrated correctness. November 13: experiments saved. November 16: code/report/results frozen. November 17–18: slides and rehearsal. November 19: presentation.

## Remaining work

CUDA selectors, CUDA sampler, reference tests, benchmark runner, and synthetic workload generator are not implemented yet. This initial commit prepares reproducible real-model inputs. Keep the proposal's synthetic grid (V = 8192, 32768, 50000, 131072; four distributions) as separate required experiments; the real-model dataset supplements it and does not cover those vocabulary sizes.

## Full-sort CUDA baseline (Kashvi)

The first implementation and CPU-reference GPU correctness test are in `src/full_sort.cu` and `tests/test_full_sort.cu`. See [build, test, and interface instructions](docs/full-sort.md). CUDA compilation and execution must be verified on CARC; this implementation has not yet been benchmarked.
