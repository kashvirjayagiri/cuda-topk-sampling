"""Validate exact radix Top-k against a global stable PyTorch sort."""
import argparse
from pathlib import Path


def reference(logits, k):
    import torch

    ids = torch.argsort(logits, dim=1, descending=True, stable=True)[:, :k]
    return logits.gather(1, ids), ids.to(torch.int32)


def check_case(logits, k, selector, expected=None, stream=None):
    import torch

    if expected is None:
        expected = reference(logits, k)
    expected_scores, expected_ids = expected
    with torch.cuda.stream(stream or torch.cuda.current_stream()):
        device_input = logits.to('cuda')
        actual_scores, actual_ids = selector(device_input)
        # The second call checks reuse of the same workspace and output buffers.
        actual_scores, actual_ids = selector(device_input)
    torch.cuda.synchronize()
    actual_ids = actual_ids.cpu()
    actual_scores = actual_scores.cpu()
    torch.testing.assert_close(actual_ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(actual_scores, expected_scores, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(device_input.cpu(), logits, rtol=0, atol=0)
    assert torch.equal(device_input.cpu().view(torch.int32), logits.view(torch.int32))
    for row in actual_ids:
        assert row.unique().numel() == k, 'Duplicate selected token ID'


def synthetic_cases():
    import torch

    generator = torch.Generator().manual_seed(451)
    for batch in (1, 8, 32):
        for vocab in (1, 2, 7, 17, 255, 256, 257, 513, 1025):
            for kind in ('negative', 'repeated', 'equal'):
                if kind == 'negative':
                    logits = -torch.rand((batch, vocab), generator=generator)
                elif kind == 'repeated':
                    logits = torch.randint(-4, 5, (batch, vocab), generator=generator).float()
                else:
                    logits = torch.full((batch, vocab), -3.0)
                for k in (1, 10, 50, 100, 257):
                    if k <= vocab:
                        yield logits, k

    adjacent = torch.nextafter(torch.tensor(1.0), torch.tensor(2.0)).item()
    special = torch.tensor([-1e30, -1.0, -adjacent, -0.0, 0.0,
                            1.0, adjacent, 1e30], dtype=torch.float32)
    for vocab in (2, 17, 257, 513):
        logits = special.repeat((vocab + len(special) - 1) // len(special))[:vocab].reshape(1, vocab)
        yield logits, vocab
    tied = torch.arange(513, dtype=torch.float32).neg_().sub_(1000)
    tied[255:257] = 100
    yield tied.reshape(1, 513), 1
    yield tied.expand(8, -1).contiguous(), 50

    for vocab in (8192, 32768, 50000, 131072):
        normal = torch.randn((1, vocab), generator=generator)
        yield normal, 100
        yield normal * 0.01, 100
        peaked = normal.clone()
        peaked[0, vocab // 2] += 15
        yield peaked, 100
        yield torch.randint(-4, 5, (1, vocab), generator=generator).float(), 100
    yield torch.randint(-4, 5, (32, 131072), generator=generator).float(), 100


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/logits/distilgpt2'))
    parser.add_argument('--library', type=Path, help='Override build/libradix_topk_python.so')
    args = parser.parse_args()

    import torch
    from radix_topk import RadixTopK

    if not torch.cuda.is_available():
        parser.error('Run inside a CARC GPU session with CUDA-enabled PyTorch.')
    files = sorted(args.data.glob('*.pt'))
    if len(files) != 10:
        parser.error(f'Expected ten committed .pt files in {args.data}.')

    cases = 0
    stream = torch.cuda.Stream()
    for logits, k in synthetic_cases():
        selector = RadixTopK(*logits.shape, k, 'cuda', args.library)
        check_case(logits, k, selector, stream=stream)
        cases += 1
    print(f'PASS: {cases} synthetic radix cases', flush=True)

    for path in files:
        payload = torch.load(path, map_location='cpu', weights_only=True)
        logits = payload['logits']
        if (logits.shape != (32, 50257) or logits.dtype != torch.float32
                or not logits.is_contiguous() or not torch.isfinite(logits).all()):
            raise ValueError(f'Invalid saved logits in {path}')
        full_scores, full_ids = reference(logits, 100)
        for batch in (1, 8, 32):
            for k in (1, 10, 50, 100):
                selector = RadixTopK(batch, 50257, k, 'cuda', args.library)
                for start in range(0, 32, batch):
                    expected = (full_scores[start:start + batch, :k],
                                full_ids[start:start + batch, :k])
                    check_case(logits[start:start + batch].contiguous(), k, selector, expected)
                    cases += 1
        print(f'PASS {path.name}: all 32 rows, global radix Top-k', flush=True)
    print(f'PASS: {cases} radix GPU cases across synthetic inputs and ten saved groups.')


if __name__ == '__main__':
    main()
