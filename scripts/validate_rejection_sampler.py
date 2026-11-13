"""Validate direct Top-k rejection samples and their small-case frequencies."""
import argparse
import math
from pathlib import Path
import statistics


def summarize(attempts):
    rejections = [count - 1 for count in attempts]
    return (f'rejections mean={statistics.mean(rejections):.3f}, '
            f'median={statistics.median(rejections):.1f}, max={max(rejections)}, '
            f'acceptance={len(attempts) / sum(attempts):.3f}')


def check_case(logits, k, temperature, sampler, draw_index, expected_ids=None,
               stream=None, replay=False):
    import torch

    batch, vocab = logits.shape
    if expected_ids is None:
        expected_ids = torch.argsort(logits, dim=1, descending=True, stable=True)[:, :k]
    device_input = logits.to('cuda')
    stream = stream or torch.cuda.current_stream()
    with torch.cuda.stream(stream):
        ids, attempts = sampler(device_input, temperature, seed=451, draw_index=draw_index)
        saved_ids, saved_attempts = ids.clone(), attempts.clone()
        if replay:
            ids, attempts = sampler(device_input, temperature, seed=451, draw_index=draw_index)
    stream.synchronize()
    actual_ids = saved_ids.cpu().tolist()
    actual_attempts = saved_attempts.cpu().tolist()
    for row in range(batch):
        if (actual_ids[row] not in expected_ids[row].tolist()
                or not 1 <= actual_attempts[row] <= vocab - k + 1):
            raise AssertionError(f'Invalid sample row={row} B={batch} V={vocab} k={k}')
        if k in (1, vocab) and actual_attempts[row] != 1:
            raise AssertionError(f'Expected one attempt for k={k}, V={vocab}')
    if replay:
        torch.testing.assert_close(ids.cpu(), saved_ids.cpu(), rtol=0, atol=0)
        torch.testing.assert_close(attempts.cpu(), saved_attempts.cpu(), rtol=0, atol=0)
    if not torch.equal(device_input.cpu().view(torch.int32), logits.view(torch.int32)):
        raise AssertionError('Input logits changed')
    return actual_attempts


def synthetic_cases():
    import torch

    generator = torch.Generator().manual_seed(451)
    for batch in (1, 8, 32):
        for vocab in (1, 3, 17, 257, 513):
            for kind in ('negative', 'repeated', 'equal'):
                if kind == 'negative':
                    logits = -torch.rand((batch, vocab), generator=generator)
                elif kind == 'repeated':
                    logits = torch.randint(-2, 3, (batch, vocab), generator=generator).float()
                else:
                    logits = torch.full((batch, vocab), -3.0)
                for k in (1, 10, 50, 100, vocab):
                    if k <= vocab:
                        yield logits, k, 1.0
    tied = torch.arange(513, dtype=torch.float32).neg_().sub_(1000)
    tied[255:257] = 100
    yield tied.reshape(1, 513), 1, 0.7
    yield tied.expand(8, -1).contiguous(), 50, 2.5
    wide = torch.tensor([-1e30, -0.0, 0.0, 1e30], dtype=torch.float32)
    yield wide.reshape(1, 4), 3, 0.1

    # Same four distributions and vocabulary sizes as generate_synthetic_logits.py.
    for vocab in (8192, 32768, 50000, 131072):
        normal = torch.randn((1, vocab), generator=generator)
        yield normal, 100, 1.0
        yield normal * 0.01, 100, 1.0
        peaked = normal.clone()
        peaked[0, vocab // 2] += 15
        yield peaked, 100, 1.0
        yield torch.randint(-4, 5, (1, vocab), generator=generator).float(), 100, 1.0
    yield torch.randint(-4, 5, (32, 131072), generator=generator).float(), 100, 1.0


def statistical_case(scores, k, temperature, sampler_class, library, name):
    import torch

    # Predeclared rule: 32,768 samples; each expected count >= 2,000; every
    # Top-k frequency must lie within six binomial standard deviations.
    rows, launches = 32, 1024
    logits = torch.tensor(scores, dtype=torch.float32).repeat(rows, 1)
    sampler = sampler_class(rows, len(scores), k, 'cuda', library)
    selected = torch.argsort(logits[0], descending=True, stable=True)[:k]
    top_scores = logits[0, selected].double()
    weights = torch.exp((top_scores - top_scores.max()) / temperature)
    probabilities = (weights / weights.sum()).tolist()
    count = rows * launches
    if min(probabilities) * count < 2000:
        raise ValueError('Statistical case has too few expected samples')

    device_input = logits.to('cuda')
    samples = torch.empty((launches, rows), device='cuda', dtype=torch.int32)
    tries = torch.empty((launches, rows), device='cuda', dtype=torch.int32)
    for draw_index in range(launches):
        ids, attempts = sampler(device_input, temperature, seed=451,
                                draw_index=draw_index + 100000)
        samples[draw_index].copy_(ids)
        tries[draw_index].copy_(attempts)
    torch.cuda.synchronize()
    observed = samples.cpu().flatten()
    attempts = tries.cpu().flatten().tolist()
    if min(attempts) < 1 or max(attempts) > len(scores) - k + 1:
        raise AssertionError(f'{name}: invalid attempt count')
    if not torch.equal(device_input.cpu().view(torch.int32), logits.view(torch.int32)):
        raise AssertionError(f'{name}: input logits changed')
    worst_z = 0.0
    for token, probability in zip(selected.tolist(), probabilities):
        frequency = (observed == token).sum().item() / count
        sigma = math.sqrt(probability * (1.0 - probability) / count)
        z = abs(frequency - probability) / sigma
        worst_z = max(worst_z, z)
        if z > 6.0:
            raise AssertionError(f'{name}: ID {token} frequency {frequency:.5f}, '
                                 f'expected {probability:.5f}, z={z:.2f} > 6')
    if not torch.isin(observed, selected.to(torch.int32)).all():
        raise AssertionError(f'{name}: sampled ID outside Top-k')
    print(f'PASS statistical {name}: N={count}, worst z={worst_z:.2f}, {summarize(attempts)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/logits/distilgpt2'))
    parser.add_argument('--library', type=Path, help='Override build/librejection_sampler_python.so')
    args = parser.parse_args()

    import torch
    from rejection_sampler import RejectionSampler

    if not torch.cuda.is_available():
        parser.error('Run inside a CARC GPU session with CUDA-enabled PyTorch.')
    files = sorted(args.data.glob('*.pt'))
    if len(files) != 10:
        parser.error(f'Expected ten committed .pt files in {args.data}.')

    cases = 0
    all_attempts = []
    stream = torch.cuda.Stream()
    for logits, k, temperature in synthetic_cases():
        sampler = RejectionSampler(*logits.shape, k, 'cuda', args.library)
        all_attempts.extend(check_case(logits, k, temperature, sampler, cases,
                                       stream=stream, replay=True))
        cases += 1
    print(f'PASS: {cases} synthetic membership cases; {summarize(all_attempts)}', flush=True)

    statistical_case([2.0, 1.0, 0.0], 3, 1.0,
                     RejectionSampler, args.library, 'one weighted draw')
    statistical_case([2.0, 1.0, 0.0, -1.0, -2.0], 3, 1.0,
                     RejectionSampler, args.library, 'weighted')
    statistical_case([0.0, 0.0, 0.0, 0.0, 0.0], 3, 1.0,
                     RejectionSampler, args.library, 'all-equal ties')

    for path in files:
        payload = torch.load(path, map_location='cpu', weights_only=True)
        logits = payload['logits']
        if (logits.shape != (32, 50257) or logits.dtype != torch.float32
                or not logits.is_contiguous() or not torch.isfinite(logits).all()):
            raise ValueError(f'Invalid saved logits in {path}')
        top_ids = torch.argsort(logits, dim=1, descending=True, stable=True)[:, :100]
        group_attempts = []
        for batch in (1, 8, 32):
            for k in (1, 10, 50, 100):
                sampler = RejectionSampler(batch, 50257, k, 'cuda', args.library)
                for start in range(0, 32, batch):
                    group_attempts.extend(check_case(
                        logits[start:start + batch].contiguous(), k, 1.0, sampler,
                        draw_index=start + batch * 1000 + k * 100000,
                        expected_ids=top_ids[start:start + batch, :k]))
                    cases += 1
        all_attempts.extend(group_attempts)
        print(f'PASS {path.name}: all 32 rows, {summarize(group_attempts)}', flush=True)
    print(f'PASS: {cases} membership cases across synthetic and ten saved groups; '
          f'{summarize(all_attempts)}')


if __name__ == '__main__':
    main()
