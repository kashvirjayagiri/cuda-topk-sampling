"""Validate Kashvi's CUDA full sort on saved full-vocabulary model logits."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/logits/distilgpt2'))
    parser.add_argument('--library', type=Path, help='Override build/libfull_sort_python.so')
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[1, 8, 32])
    parser.add_argument('--k', nargs='+', type=int, default=[1, 10, 50, 100])
    args = parser.parse_args()

    import torch
    from full_sort import FullSort

    if not torch.cuda.is_available():
        parser.error('Run inside a CARC GPU session with CUDA-enabled PyTorch.')
    files = sorted(args.data.glob('*.pt'))
    if not files:
        parser.error(f'No .pt inputs in {args.data}. Pull the committed dataset first.')
    if any(b <= 0 for b in args.batch_sizes) or any(k <= 0 for k in args.k):
        parser.error('Batch sizes and k must be positive.')

    cases = 0
    for path in files:
        payload = torch.load(path, map_location='cpu', weights_only=True)
        logits = payload['logits']
        if (logits.ndim != 2 or logits.dtype != torch.float32
                or not torch.isfinite(logits).all() or logits.shape[0] == 0):
            raise ValueError(f'Invalid input tensor in {path}')
        rows, vocab_size = logits.shape
        # CPU reference: original columns are ascending IDs, and stable sorting
        # preserves that order when scores tie. Compute it once per group.
        expected_ids = torch.argsort(logits, descending=True, stable=True, dim=1)
        for batch_size in args.batch_sizes:
            if batch_size > rows or rows % batch_size:
                raise ValueError(f'{path}: B={batch_size} must divide the {rows} rows')
            for k in args.k:
                if k > vocab_size:
                    raise ValueError(f'k={k} exceeds V={vocab_size}')
                sorter = FullSort(batch_size, vocab_size, k, 'cuda', args.library)
                # Cover every row, not just the first batch in each topic group.
                for start in range(0, rows, batch_size):
                    original = logits[start:start + batch_size].contiguous()
                    device_input = original.to('cuda')
                    actual_scores, actual_ids = sorter(device_input)
                    torch.cuda.synchronize()  # Also expose asynchronous execution errors.
                    ids = expected_ids[start:start + batch_size, :k]
                    torch.testing.assert_close(actual_ids.cpu().long(), ids, rtol=0, atol=0)
                    torch.testing.assert_close(actual_scores.cpu(), original.gather(1, ids),
                                               rtol=1e-5, atol=1e-5)
                    torch.testing.assert_close(device_input.cpu(), original, rtol=0, atol=0)
                    cases += 1
        print(f'PASS {path.name}: {rows} rows, V={vocab_size}', flush=True)
    print(f'PASS: {cases} GPU sorting calls checked across {len(files)} files.')
    print('Correctness only: this runtime is not a kernel performance benchmark.')


if __name__ == '__main__':
    main()
