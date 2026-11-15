"""Validate hierarchical Top-k plus parallel sampling on synthetic and saved logits."""
import argparse
from pathlib import Path


def reference(logits, k, temperature):
    import torch

    ids = torch.argsort(logits, dim=1, descending=True, stable=True)[:, :k]
    scores = logits.gather(1, ids)
    shifted = scores.double() - scores.double().max(dim=1, keepdim=True).values
    weights = torch.exp(shifted / temperature)
    probabilities = weights / weights.sum(dim=1, keepdim=True)
    cdf = torch.cumsum(probabilities, dim=1).clamp(max=1.0)
    cdf[:, -1] = 1.0
    return scores, ids.to(torch.int32), cdf


def randoms_away_from_boundaries(cdf, generator):
    """Use supplied FP32 uniforms that expose ordering without CDF-rounding ties."""
    import torch

    batch = cdf.shape[0]
    randoms = torch.empty(batch, dtype=torch.float32)
    for row in range(batch):
        for _ in range(1000):
            value = torch.rand((), generator=generator, dtype=torch.float32)
            if torch.min(torch.abs(cdf[row] - value.double())) > 5e-5:
                randoms[row] = value
                break
        else:
            raise RuntimeError('Could not choose a uniform away from CDF boundaries')
    return randoms


def check_case(logits, k, temperature, runner, generator, stream=None):
    import torch

    expected_scores, expected_ids, cdf = reference(logits, k, temperature)
    randoms = randoms_away_from_boundaries(cdf, generator)
    positions = torch.searchsorted(cdf, randoms.double().unsqueeze(1).contiguous(),
                                   right=True)
    expected_samples = expected_ids.gather(1, positions).squeeze(1)
    device_input = logits.to('cuda')
    device_randoms = randoms.to('cuda')
    stream = stream or torch.cuda.current_stream()
    stream.wait_stream(torch.cuda.current_stream(device_input.device))
    with torch.cuda.stream(stream):
        actual_scores, actual_ids, actual_samples = runner(
            device_input, device_randoms, temperature)
    stream.synchronize()
    torch.testing.assert_close(actual_ids.cpu(), expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(actual_scores.cpu(), expected_scores, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_samples.cpu(), expected_samples, rtol=0, atol=0)
    torch.testing.assert_close(device_input.cpu(), logits, rtol=0, atol=0)


def synthetic_cases():
    import torch

    generator = torch.Generator().manual_seed(451)
    yield torch.tensor([[1, -2, 3, -5, 2], [-4, 0, 0, -3, 0]], dtype=torch.float32), 3, 1.0
    for batch in (1, 8, 32):
        for vocab in (17, 257, 259, 513, 1025):
            for kind in ('negative', 'repeated', 'equal'):
                if kind == 'negative':
                    logits = -torch.rand((batch, vocab), generator=generator)
                elif kind == 'repeated':
                    logits = torch.randint(-2, 3, (batch, vocab), generator=generator).float()
                else:
                    logits = torch.full((batch, vocab), -3.0)
                for k in (1, 10, 50, 100):
                    if k <= vocab:
                        yield logits, k, 1.0
    tied = (-1000 - torch.arange(512)).float()
    tied[255:257] = 100
    yield tied.reshape(1, 512), 1, 0.7
    yield tied.expand(8, -1).contiguous(), 10, 2.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/logits/distilgpt2'))
    parser.add_argument('--library', type=Path,
                        help='Override build/libhierarchical_topk_sampling_python.so')
    args = parser.parse_args()

    import torch
    from hierarchical_topk_sampling import HierarchicalTopKSampling

    if not torch.cuda.is_available():
        parser.error('Run inside a CARC GPU session with CUDA-enabled PyTorch.')
    files = sorted(args.data.glob('*.pt'))
    if len(files) != 10:
        parser.error(f'Expected ten committed .pt files in {args.data}.')

    generator = torch.Generator().manual_seed(451)
    stream = torch.cuda.Stream()
    cases = 0
    for logits, k, temperature in synthetic_cases():
        runner = HierarchicalTopKSampling(*logits.shape, k, 'cuda', args.library)
        check_case(logits, k, temperature, runner, generator, stream)
        cases += 1
    print(f'PASS: {cases} synthetic combined cases', flush=True)

    for path in files:
        payload = torch.load(path, map_location='cpu', weights_only=True)
        logits = payload['logits']
        if (logits.shape != (32, 50257) or logits.dtype != torch.float32
                or not logits.is_contiguous() or not torch.isfinite(logits).all()):
            raise ValueError(f'Invalid saved logits in {path}')
        for batch in (1, 8, 32):
            for k in (1, 10, 50, 100):
                runner = HierarchicalTopKSampling(batch, 50257, k, 'cuda', args.library)
                for start in range(0, 32, batch):
                    check_case(logits[start:start + batch].contiguous(), k, 1.0,
                               runner, generator)
                    cases += 1
        print(f'PASS {path.name}: all 32 rows', flush=True)
    print(f'PASS: {cases} combined GPU calls across synthetic cases and ten saved groups.')


if __name__ == '__main__':
    main()
