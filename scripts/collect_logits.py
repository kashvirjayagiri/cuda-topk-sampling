"""Collect one full next-token logit vector per unpadded prompt."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompts', type=Path, default=Path('data/prompts/groups.json'))
    parser.add_argument('--output', type=Path, default=Path('data/logits/distilgpt2'))
    parser.add_argument('--model', default='distilbert/distilgpt2')
    parser.add_argument('--revision', default='main', help='Use a commit hash to repeat a prior collection.')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--groups', nargs='+', help='Optional group IDs to collect.')
    args = parser.parse_args()

    # Import after argument parsing so --help works before dependencies are installed.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    raw = args.prompts.read_bytes()
    groups = json.loads(raw)
    ids = [group['id'] for group in groups]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate group IDs')
    if args.groups:
        unknown = set(args.groups) - set(ids)
        if unknown:
            parser.error(f'Unknown groups: {sorted(unknown)}')
        groups = [g for g in groups if g['id'] in args.groups]
    for group in groups:
        if Path(group['id']).name != group['id'] or group['id'] in ('.', '..'):
            raise ValueError('Group IDs must be plain filenames')
        prompts = group['prompts']
        if not prompts or any(not isinstance(p, str) or not p.strip() for p in prompts):
            raise ValueError(f"Invalid prompts in {group['id']}")
        if len(prompts) != len(set(prompts)):
            raise ValueError(f"Duplicate prompts in {group['id']}")
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Run inside a CARC GPU allocation, or use --device cpu.')

    args.output.mkdir(parents=True, exist_ok=True)
    targets = [args.output / f"{g['id']}.pt" for g in groups]
    targets.append(args.output / 'manifest.json')
    if any(p.exists() for p in targets):
        raise FileExistsError('Output already exists. Choose a new --output directory; existing data is preserved.')

    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision)
    resolved = getattr(model.config, '_commit_hash', None)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=resolved or args.revision)
    model = model.float().to(args.device).eval()
    manifest = {
        'model': args.model, 'requested_revision': args.revision,
        'resolved_revision': resolved, 'vocab_size': model.config.vocab_size,
        'dtype': 'float32', 'device': args.device,
        'gpu': torch.cuda.get_device_name() if args.device == 'cuda' else None,
        'python': platform.python_version(), 'torch': str(torch.__version__),
        'transformers': importlib.metadata.version('transformers'),
        'cuda': torch.version.cuda,
        'prompt_file_sha256': hashlib.sha256(raw).hexdigest(),
        'collection': 'One raw next-token vector per prompt; no temperature, softmax, or filtering.',
        'groups': [],
    }
    with torch.inference_mode():
        for group in groups:
            rows, lengths = [], []
            for prompt in group['prompts']:
                encoded = tokenizer(prompt, return_tensors='pt').to(args.device)
                length = encoded['input_ids'].shape[1]
                if length > model.config.max_position_embeddings:
                    raise ValueError(f"Prompt exceeds context limit in {group['id']}")
                row = model(**encoded, use_cache=False).logits[0, -1, :].float().cpu()
                if row.shape != (model.config.vocab_size,) or not torch.isfinite(row).all():
                    raise ValueError('Unexpected shape or nonfinite logits')
                rows.append(row)
                lengths.append(length)
            logits = torch.stack(rows).contiguous()
            payload = {'logits': logits, 'prompts': group['prompts'], 'token_lengths': lengths,
                       'group_id': group['id'], 'model': args.model,
                       'model_revision': resolved, 'vocab_size': model.config.vocab_size}
            path = args.output / f"{group['id']}.pt"
            temp = path.with_suffix('.pt.tmp')
            torch.save(payload, temp)
            temp.replace(path)
            manifest['groups'].append({'id': group['id'], 'file': path.name,
                                       'shape': list(logits.shape),
                                       'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            print(f'{path}: {tuple(logits.shape)}', flush=True)
    temp = args.output / 'manifest.json.tmp'
    temp.write_text(json.dumps(manifest, indent=2) + '\n')
    temp.replace(args.output / 'manifest.json')
    print('Collection complete. Model execution must be excluded from kernel benchmarks.')


if __name__ == '__main__':
    main()
