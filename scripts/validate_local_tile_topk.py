"""Validate every local tile against a stable PyTorch sort, including saved logits."""
import argparse
from pathlib import Path


def tile_reference(logits, k):
    """Stable descending score order retains ascending original IDs on ties."""
    import torch

    batch, vocab = logits.shape
    tile_count = (vocab + 255) // 256
    padded_vocab = tile_count * 256
    scores = torch.full((batch, padded_vocab), -torch.inf, dtype=torch.float32)
    scores[:, :vocab] = logits
    ids = torch.arange(padded_vocab, dtype=torch.int32).expand(batch, -1).clone()
    ids[:, vocab:] = torch.iinfo(torch.int32).max
    scores = scores.reshape(batch, tile_count, 256)
    ids = ids.reshape(batch, tile_count, 256)
    order = torch.argsort(scores, dim=2, descending=True, stable=True)[:, :, :k]
    return scores.gather(2, order), ids.gather(2, order)


def check_case(logits, k, selector):
    import torch

    expected_scores, expected_ids = tile_reference(logits, k)
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
        for vocab in (1, 17, 250, 256, 257, 259, 512, 513):
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/logits/distilgpt2'))
    parser.add_argument('--library', type=Path, help='Override build/liblocal_tile_topk_python.so')
    args = parser.parse_args()

    import torch
    from local_tile_topk import LocalTileTopK

    if not torch.cuda.is_available():
        parser.error('Run inside a CARC GPU session with CUDA-enabled PyTorch.')
    files = sorted(args.data.glob('*.pt'))
    if len(files) != 10:
        parser.error(f'Expected ten committed .pt files in {args.data}.')

    cases = 0
    for logits, k in synthetic_cases():
        selector = LocalTileTopK(*logits.shape, k, 'cuda', args.library)
        check_case(logits, k, selector)
        cases += 1
    print(f'PASS: {cases} synthetic local-output cases', flush=True)

    for path in files:
        payload = torch.load(path, map_location='cpu', weights_only=True)
        logits = payload['logits']
        if (logits.shape != (32, 50257) or logits.dtype != torch.float32
                or not logits.is_contiguous() or not torch.isfinite(logits).all()):
            raise ValueError(f'Invalid saved logits in {path}')
        for batch in (1, 8, 32):
            for k in (1, 10, 50, 100):
                selector = LocalTileTopK(batch, logits.shape[1], k, 'cuda', args.library)
                for start in range(0, 32, batch):
                    check_case(logits[start:start + batch].contiguous(), k, selector)
                    cases += 1
        print(f'PASS {path.name}: all 32 rows, all tiles', flush=True)
    print(f'PASS: {cases} local tile GPU calls across synthetic cases and ten saved groups.')


if __name__ == '__main__':
    main()
