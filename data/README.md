# DistilGPT2 next-token logit dataset

## Teammates: start here on CARC

**The dataset has already been collected and pushed. Do not regenerate it for normal development or benchmarking.** Use the committed files so everyone compares identical scores. You do not need to download DistilGPT2, install Transformers, or run `collect_logits.py` to use these files. The regeneration section below is optional reference only.

1. Launch a CARC OnDemand Code Server session with one GPU, using your course project account and an allowed GPU partition. Wait until it is running, connect, and open its Bash terminal.
2. If you do not have the repository on CARC yet:

   ```bash
   git clone https://github.com/kashvirjayagiri/cuda-topk-sampling.git
   cd cuda-topk-sampling
   ```

   Otherwise, enter your existing repository directory and run `git pull`. Resolve any local-change warnings before proceeding.
3. Activate your own Python environment with PyTorch. If you created the environment described below, use `source ~/ee451-env/bin/activate`. Each teammate has their own home directory and environment; Kashvi's environment is not automatically available to others. For loading saved data, skip the Transformers installation line in the environment section.
4. From the repository root, check the saved data and an actual GPU operation:

   ```bash
   python - <<'PYTHON'
   from pathlib import Path
   import torch

   root = Path('data/logits/distilgpt2')
   assert len(list(root.glob('*.pt'))) == 10, 'Expected ten group files; check your branch and git pull.'
   assert (root / 'manifest.json').is_file(), 'Missing dataset manifest.'
   data = torch.load(root / '01_everyday.pt', map_location='cpu', weights_only=True)
   scores = data['logits']
   assert scores.shape == (32, 50257)
   assert scores.dtype == torch.float32 and torch.isfinite(scores).all()
   print('First prompt:', data['prompts'][0])
   print('All prompts in this file:', data['prompts'])
   scores = scores.to('cuda')
   print('GPU:', torch.cuda.get_device_name(0))
   print('Input shape:', tuple(scores.shape))
   print('Example GPU calculation:', scores[0].max().item())
   PYTHON
   ```

Expected shape: `(32, 50257)`. Each row corresponds to the prompt at the same position in `data['prompts']`; each column is an original token ID. Use the loading examples below to prepare batches for your implementation. You do not feed the prompt strings into the CUDA algorithms; you feed the saved score tensors.

If files are missing after pulling the correct branch, ask the team to confirm the dataset commit instead of collecting a different copy. If CUDA is unavailable, verify that your terminal belongs to the allocated GPU session rather than the login node. If a P100 reports an unsupported GPU architecture, check that your PyTorch build matches the tested CUDA 12.6 setup below.

## What this data is

This dataset supplies inputs for our EE 451 CUDA Top-k selection and token-sampling experiments. It contains **real model outputs**, not synthetic random scores and not training data.

A prompt is a short text prefix, such as `An introduction to making breakfast explains that`. DistilGPT2 reads the prefix and produces one raw next-token score (a **logit**) for each of its 50,257 vocabulary entries. Higher scores indicate more favored candidates. Logits are not probabilities; they can be negative and do not sum to one.

We save the complete next-token score vector for each prompt. We do not generate a paragraph, train the model, or apply temperature, softmax, or Top-k filtering during collection.

```text
10 topic groups
  32 prompts per group
    50,257 next-token scores per prompt
```

Each group contains a `[32, 50257]` contiguous FP32 tensor: about 6.4 MB. All ten groups contain 320 prompts and 16,082,240 scores: about 64.3 MB of raw tensor data, plus metadata. The downloaded model is separate and is not part of the dataset.

## Groups

The exact prompts are versioned in [prompts/groups.json](prompts/groups.json).

| File | Topic | Example subject |
|---|---|---|
| `01_everyday.pt` | Everyday activities | Making breakfast |
| `02_science.pt` | Science | Photosynthesis |
| `03_math.pt` | Mathematics | Fractions |
| `04_code.pt` | Programming explanations | A Python loop |
| `05_history.pt` | History | The printing press |
| `06_story.pt` | Story openings | A lost astronaut |
| `07_dialogue.pt` | Dialogue | Meeting a new classmate |
| `08_facts.pt` | Factual questions | The capital of France |
| `09_procedures.pt` | Procedures | Brewing tea |
| `10_engineering.pt` | Engineering | Memory bandwidth |

These are short, templated English prompts. The categories describe subjects, not guaranteed score distributions. This is a small controlled dataset from one model, not a representative benchmark of all LLM outputs. The proposal's synthetic normal, nearly uniform, peaked, and tie-heavy workloads remain separate experiments.

## Get the saved data

Run all commands below from the repository root. If the dataset has been pushed to GitHub, teammates only need:

```bash
git pull
ls data/logits/distilgpt2/
```

Expected contents: ten `.pt` files and `manifest.json`. A fresh clone includes committed dataset files automatically. If the folder is missing, confirm the data commit was pushed to your branch or regenerate it below. You do not need Transformers or the model download just to load saved tensors; you need PyTorch.

## CARC Python environment

Our initial successful collection setup used a Tesla P100, Python 3.11, and PyTorch `2.6.0+cu126`. For that setup, inside an allocated GPU session:

```bash
python -m venv ~/ee451-env
source ~/ee451-env/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install 'transformers>=4.45,<5' 'safetensors>=0.4'
python -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name(0)); print(torch.randn(100, device='cuda').sum().item())"
```

If the environment already exists, just activate it. Do not recreate it for every session. The final command checks an actual GPU operation. Consult `manifest.json` for the environment used for the saved dataset. Custom CUDA compilation also needs a compatible CUDA toolkit/compiler; installing PyTorch alone does not install `nvcc`.

## Load and inspect one group

This example runs on CPU too. It loads a saved file; it does not run DistilGPT2.

```python
import torch

path = 'data/logits/distilgpt2/01_everyday.pt'
data = torch.load(path, map_location='cpu', weights_only=True)
logits = data['logits']

assert logits.shape == (32, 50257)
assert logits.dtype == torch.float32
assert logits.is_contiguous()
assert torch.isfinite(logits).all()
print('Group:', data['group_id'])
print('First prompt:', data['prompts'][0])
print('Shape:', tuple(logits.shape))

# A stable descending sort preserves ascending original IDs for equal scores.
k = 10
ids = torch.argsort(logits, dim=-1, descending=True, stable=True)[:, :k]
values = logits.gather(1, ids)
print('First prompt, top IDs:', ids[0].tolist())
print('First prompt, top scores:', values[0].tolist())
# PyTorch indices are int64; convert to int32 when passing to our CUDA interface.
```

Each `.pt` file is a dictionary:

| Key | Contents |
|---|---|
| `logits` | CPU FP32 tensor `[32, 50257]` |
| `prompts` | 32 input strings, in tensor-row order |
| `token_lengths` | Tokenized length of each prompt |
| `group_id` | Group filename stem |
| `model` | Hugging Face model identifier |
| `model_revision` | Resolved model commit, when available |
| `vocab_size` | Number of vocabulary entries |

Column j is original token ID j. Rows are independent next-token decisions, not consecutive tokens in one generated sequence.

## Prepare GPU inputs for the algorithms

```python
import torch

data = torch.load('data/logits/distilgpt2/01_everyday.pt', weights_only=True)
B, k = 8, 10
scores = data['logits'][:B].contiguous().to('cuda')
g = torch.Generator(device='cuda').manual_seed(451)
u = torch.rand(B, generator=g, device='cuda')
# Pass the SAME scores to both selectors and PyTorch.
# Pass the SAME u to their common sampling stage.
```

Use batch sizes `1, 8, 32` and k values `1, 10, 50, 100`. Rotate through row windows for smaller batches and test all groups. For this real dataset, report **V = 50,257**; do not trim to 50,000. Full bitonic sort pads internally to 65,536, but padded entries must not appear in selected results.

The project CUDA selectors and sampler are separate implementations; these loading examples do not run those kernels. Measure selection and sampling only, with inputs/buffers already on the GPU. Exclude model execution, disk reads, transfers, and allocation. Topic groups are different workloads, not substitutes for repeated timing measurements.

## Regenerate the data (optional)

Use the saved files for comparisons so everyone uses identical inputs. Regeneration needs Transformers, network access for the first model download, and a GPU allocation by default.

One-group smoke test:

```bash
python scripts/collect_logits.py --groups 01_everyday --output data/logits/smoke
```

All ten groups, if the default output does not already exist:

```bash
python scripts/collect_logits.py
```

The collector refuses to overwrite files. If data is already present, choose a new output directory. To repeat the recorded model revision, replace `COMMIT_FROM_MANIFEST` with `resolved_revision` from the existing manifest:

```bash
python scripts/collect_logits.py --revision COMMIT_FROM_MANIFEST --output data/logits/recollected
```

Use `--device cpu` for CPU-only collection if necessary; this is not GPU benchmarking. If a run is interrupted, collect again into a fresh directory. Different hardware/software can produce small numerical differences even with the same model revision.

`manifest.json` records the model revision, versions, collection device, prompt-file SHA-256, group shapes, and SHA-256 hashes of saved files. Keep it with the tensors.

## Share through GitHub

Generated outputs are ignored by default to avoid accidental uploads. To deliberately commit this fixed dataset:

```bash
git add -f data/logits/distilgpt2/*.pt data/logits/distilgpt2/manifest.json
git diff --cached --stat
# Confirm ten .pt files and one manifest; review any other staged files too.
git commit -m "Add ten groups of DistilGPT2 logits"
git push
```

Do not add the model cache or smoke-test outputs. Replacing binary datasets repeatedly increases Git history size. Teammates get the committed data with `git pull`.

## Sources

- [DistilGPT2 model](https://huggingface.co/distilbert/distilgpt2)
- [DistilGPT2 vocabulary configuration](https://huggingface.co/distilbert/distilgpt2/blob/main/config.json)
- [Hugging Face model output documentation](https://huggingface.co/docs/transformers/main_classes/output)
