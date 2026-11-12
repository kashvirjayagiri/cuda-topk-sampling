"""Validate global hierarchical Top-k against full-row stable PyTorch sorting."""
import argparse
from pathlib import Path


def global_reference(logits, k):
    import torch

    ids = torch.argsort(logits, dim=1, descending=True, stable=True)[:, :k]
    return logits.gather(1, ids), ids.to(torch.int32)


def check_case(logits, k, selector, expected=None):
    import torch

    expected_scores, expected_ids = global_reference(logits, k) if expected is None else expected
    device_input = logits.to('cuda')
    actual_scores, actual_ids = selector(device_input)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual_ids.cpu(), expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(actual_scores.cpu(), expected_scores, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(device_input.cpu(), logits, rtol=0, atol=0)


def synthetic_cases():
    import torch

    generator = torch.Generator().manual_seed(451)
    for batch in (1, 8, 32):
        for vocab in (1, 17, 250, 256, 257, 259, 512, 513, 1025):
            for kind in ('negative', 'repeated', 'equal'):
                if kind == 'negative':
                    logits = -torch.rand((batch, vocab), generator=generator)
                elif kind == 'repeated':
                    logits = torch.randint(-2, 3, (batch, vocab), generator=generator).float()
                else:
                    logits = torch.full((batch, vocab), -3.0)
                for k in (1, 10, 50, 100):
                    if k <= vocab:
                        yield logits, k
    # Adjacent tiles have interleaved winners.
    ids = torch.arange(512)
    yield (1000 - 2 * (ids % 256) - (ids // 256)).float().reshape(1, 512), 100
    # The highest score occurs at IDs 255 and 256 across a tile boundary.
    tied = (-1000 - ids).float()
    tied[255:257] = 1000
    yield tied.reshape(1, 512), 1
    yield tied.expand(8, -1).contiguous(), 100
    for vocab in (8192, 32768, 50000, 131072):
        yield -torch.rand((1, vocab), generator=generator), 100
    yield torch.randint(-2, 3, (32, 131072), generator=generator).float(), 100


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/logits/distilgpt2'))
    parser.add_argument('--library', type=Path, help='Override build/libhierarchical_topk_python.so')
    args = parser.parse_args()

    import torch
    from hierarchical_topk import HierarchicalTopK

    if not torch.cuda.is_available():
        parser.error('Run inside a CARC GPU session with CUDA-enabled PyTorch.')
    files = sorted(args.data.glob('*.pt'))
    if len(files) != 10:
        parser.error(f'Expected ten committed .pt files in {args.data}.')

    cases = 0
    for logits, k in synthetic_cases():
        selector = HierarchicalTopK(*logits.shape, k, 'cuda', args.library)
        check_case(logits, k, selector)
        cases += 1
    print(f'PASS: {cases} synthetic global-output cases', flush=True)

    for path in files:
        payload = torch.load(path, map_location='cpu', weights_only=True)
        logits = payload['logits']
        if (logits.shape != (32, 50257) or logits.dtype != torch.float32
                or not logits.is_contiguous() or not torch.isfinite(logits).all()):
            raise ValueError(f'Invalid saved logits in {path}')
        expected_scores, expected_ids = global_reference(logits, 100)
        for batch in (1, 8, 32):
            for k in (1, 10, 50, 100):
                selector = HierarchicalTopK(batch, logits.shape[1], k, 'cuda', args.library)
                for start in range(0, 32, batch):
                    expected = (expected_scores[start:start + batch, :k],
                                expected_ids[start:start + batch, :k])
                    check_case(logits[start:start + batch].contiguous(), k, selector, expected)
                    cases += 1
        print(f'PASS {path.name}: all 32 rows, global Top-k', flush=True)
    print(f'PASS: {cases} hierarchical GPU calls across synthetic cases and ten saved groups.')


if __name__ == '__main__':
    main()
