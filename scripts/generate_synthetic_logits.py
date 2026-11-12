"""Generate shared FP32 synthetic workloads for the selection benchmarks."""

import argparse
import json
from pathlib import Path


VOCAB_SIZES = (8192, 32768, 50000, 131072)
DISTRIBUTIONS = {
    'normal': 'Standard normal logits',
    'near_uniform': 'Normal logits scaled by 0.01 for nearly uniform softmax',
    'peaked': 'Standard normal logits with one randomly chosen token boosted by 15 per row',
    'tie_heavy': 'Uniform integer scores from -4 through 4, converted to FP32',
}
ROWS = 32


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data/logits/synthetic'))
    parser.add_argument('--seed', type=int, default=451)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f'{args.output} already exists. Choose a fresh output directory.')

    import torch

    generator = torch.Generator(device='cpu').manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    for vocab in VOCAB_SIZES:
        for distribution in DISTRIBUTIONS:
            if distribution == 'tie_heavy':
                logits = torch.randint(-4, 5, (ROWS, vocab), generator=generator,
                                       device='cpu').to(torch.float32)
            else:
                logits = torch.randn(ROWS, vocab, generator=generator,
                                     device='cpu', dtype=torch.float32)
                if distribution == 'near_uniform':
                    logits.mul_(0.01)
                elif distribution == 'peaked':
                    winners = torch.randint(vocab, (ROWS,), generator=generator, device='cpu')
                    logits[torch.arange(ROWS, device='cpu'), winners] += 15.0

            path = args.output / f'v{vocab}_{distribution}.pt'
            torch.save({'logits': logits, 'distribution': distribution}, path)
            print(f'{path}: {tuple(logits.shape)}')

    manifest = {
        'kind': 'synthetic',
        'seed': args.seed,
        'torch_version': str(torch.__version__),
        'rows_per_file': ROWS,
        'dtype': 'float32',
        'vocab_sizes': list(VOCAB_SIZES),
        'distributions': DISTRIBUTIONS,
    }
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Generated {len(VOCAB_SIZES) * len(DISTRIBUTIONS)} workloads in {args.output}')


if __name__ == '__main__':
    main()
