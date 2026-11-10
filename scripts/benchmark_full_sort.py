"""Measure full-sort selection latency and explicit workspace on saved logits."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command_output(*args):
    try:
        return subprocess.check_output(list(args), text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/logits/distilgpt2'))
    parser.add_argument('--groups', nargs='+', help='Filename stems, e.g. 01_everyday')
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[1, 8, 32])
    parser.add_argument('--k', nargs='+', type=int, default=[1, 10, 50, 100])
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--repetitions', type=int, default=50)
    parser.add_argument('--library', type=Path, default=Path('build/libfull_sort_python.so'))
    parser.add_argument('--output', type=Path, help='New output directory; defaults to timestamp under results/')
    args = parser.parse_args()
    if args.warmup < 1 or args.repetitions < 1:
        parser.error('Warmup and repetitions must be positive.')
    if any(n < 1 for n in args.batch_sizes + args.k):
        parser.error('Batch sizes and k must be positive.')
    files = sorted(args.data.glob('*.pt'))
    if args.groups:
        unknown = set(args.groups) - {p.stem for p in files}
        if unknown:
            parser.error(f'Unknown groups: {sorted(unknown)}')
        files = [p for p in files if p.stem in args.groups]
    if not files:
        parser.error('No saved data found. Pull the committed dataset first.')
    if not args.library.is_file():
        parser.error('Shared library missing; build the CMake targets first.')

    import torch
    from full_sort import FullSort

    if not torch.cuda.is_available():
        parser.error('Run inside an allocated CARC GPU session.')
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = args.output or Path('results') / f'full-sort-{timestamp}'
    output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[1]
    metadata = {
        'started_utc': timestamp, 'status': 'running',
        'method': 'full_sort', 'timing': 'native CUDA events around one full selection call',
        'sampling_included': False, 'dtype': 'float32',
        'gpu': torch.cuda.get_device_name(0),
        'compute_capability': list(torch.cuda.get_device_capability(0)),
        'torch': str(torch.__version__), 'cuda_runtime': torch.version.cuda,
        'git_commit': command_output('git', 'rev-parse', 'HEAD'),
        'git_worktree_status': command_output('git', 'status', '--porcelain'),
        'nvcc_version': command_output('nvcc', '--version'),
        'driver_versions': command_output('nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'),
        'warmup': args.warmup, 'repetitions': args.repetitions,
        'batch_sizes': args.batch_sizes, 'k': args.k,
        'library': str(args.library.resolve()), 'library_sha256': file_hash(args.library),
        'source_sha256': {p: file_hash(repo / p) for p in [
            'src/full_sort.cu', 'src/full_sort_api.cu', 'include/topk.cuh',
            'scripts/full_sort.py', 'scripts/benchmark_full_sort.py']},
        'inputs_sha256': {p.name: file_hash(p) for p in files},
        'memory_scope': 'Explicit tensor bytes, not allocator reservation or process peak',
        'excluded': ['disk reads', 'host/device copies', 'allocation', 'validation', 'model inference'],
    }
    manifest = args.data / 'manifest.json'
    if manifest.exists():
        metadata['dataset_manifest'] = json.loads(manifest.read_text())
    meta_path = output / 'metadata.json'
    meta_path.write_text(json.dumps(metadata, indent=2) + '\n')

    fields = ['group', 'B', 'V', 'k', 'row_start', 'warmup', 'repetitions',
              'median_ms', 'min_ms', 'max_ms', 'rows_per_second',
              'workspace_bytes', 'input_bytes', 'output_bytes', 'total_tensor_bytes']
    try:
        with (output / 'summary.csv').open('w', newline='') as summary_file, \
             (output / 'samples.csv').open('w', newline='') as samples_file:
            summary = csv.DictWriter(summary_file, fieldnames=fields)
            samples = csv.writer(samples_file)
            summary.writeheader()
            samples.writerow(['group', 'B', 'V', 'k', 'row_start', 'repeat', 'milliseconds'])
            for path in files:
                data = torch.load(path, map_location='cpu', weights_only=True)
                logits = data['logits']
                if (logits.ndim != 2 or logits.dtype != torch.float32
                        or not torch.isfinite(logits).all() or logits.shape[0] == 0):
                    raise ValueError(f'Invalid tensor in {path}')
                rows, vocab = logits.shape
                # Compute an independent, tie-broken answer outside timing.
                expected = torch.argsort(logits, dim=1, descending=True, stable=True)
                for batch in args.batch_sizes:
                    if rows % batch or batch > rows:
                        raise ValueError(f'B={batch} must divide {rows} rows in {path}')
                    for k in args.k:
                        sorter = FullSort(batch, vocab, k, 'cuda', args.library)
                        medians = []
                        for start in range(0, rows, batch):
                            original = logits[start:start + batch].contiguous()
                            scores = original.to('cuda')
                            values, ids = sorter(scores)
                            reference = expected[start:start + batch, :k]
                            torch.testing.assert_close(ids.cpu().long(), reference, rtol=0, atol=0)
                            torch.testing.assert_close(values.cpu(), original.gather(1, reference),
                                                       rtol=1e-5, atol=1e-5)
                            timings = sorter.benchmark(scores, args.warmup, args.repetitions)
                            if any(t <= 0 for t in timings):
                                raise RuntimeError('Nonpositive timing; inspect the CUDA event setup')
                            # Verify repeated execution also preserves results/input.
                            torch.testing.assert_close(ids.cpu().long(), reference, rtol=0, atol=0)
                            torch.testing.assert_close(values.cpu(), original.gather(1, reference),
                                                       rtol=1e-5, atol=1e-5)
                            torch.testing.assert_close(scores.cpu(), original, rtol=0, atol=0)
                            median = statistics.median(timings)
                            medians.append(median)
                            input_bytes = scores.nbytes
                            output_bytes = values.nbytes + ids.nbytes
                            summary.writerow(dict(zip(fields, [
                                path.stem, batch, vocab, k, start, args.warmup, args.repetitions,
                                median, min(timings), max(timings), batch * 1000 / median,
                                sorter.workspace_bytes, input_bytes, output_bytes,
                                sorter.workspace_bytes + input_bytes + output_bytes])))
                            for index, timing in enumerate(timings):
                                samples.writerow([path.stem, batch, vocab, k, start, index, timing])
                        summary_file.flush()
                        samples_file.flush()
                        print(f'{path.stem} B={batch} k={k}: '
                              f'median of window medians={statistics.median(medians):.4f} ms; '
                              f'workspace={sorter.workspace_bytes / 2**20:.2f} MiB', flush=True)
    except BaseException as error:
        metadata['status'] = 'incomplete'
        metadata['error'] = str(error)
        raise
    else:
        metadata['status'] = 'complete'
    finally:
        meta_path.write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Results saved to {output}')


if __name__ == '__main__':
    main()
