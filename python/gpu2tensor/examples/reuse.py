"""Compare whole-request cost on fresh and reused processes on the same GPU."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from gpu2tensor import Candidate, Evaluator
from gpu2tensor.examples.softmax import sources, workload


def check_result(result, mode):
    assert result.correct and result.record['profile_status'] == 'ok', result.record
    reused = result.record.get('process', {}).get('mode') == 'reused'
    assert reused == (mode == 'reused'), 'Endpoint has the wrong process mode.'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fresh', required=True)
    parser.add_argument('--reused', required=True)
    parser.add_argument('--trials', type=int, default=4)
    parser.add_argument('--output', type=Path, default=Path('artifacts/reuse'))
    args = parser.parse_args()
    if args.trials < 2:
        parser.error('Use at least two paired trials.')
    workers = {'fresh': Evaluator(args.fresh), 'reused': Evaluator(args.reused)}
    candidate = Candidate(sources()[1], 'triton', 'fused_softmax')
    target = workload()
    rows = []
    for mode, worker in workers.items():
        result = worker.evaluate(candidate, target, profile=True, repetitions=30)
        result.save(args.output / (mode + '-warmup'))
        check_result(result, mode)
    policy = None
    for trial in range(args.trials):
        order = ('fresh', 'reused') if trial % 2 == 0 else ('reused', 'fresh')
        for mode in order:
            started = time.perf_counter()
            result = workers[mode].evaluate(candidate, target, profile=True, repetitions=30)
            elapsed = time.perf_counter() - started
            result.save(args.output / f'{mode}-{trial}')
            check_result(result, mode)
            identity = {key: result.record[key] for key in
                        ('host', 'device', 'inputs', 'timing_method', 'measurement_policy')}
            if policy is None:
                policy = identity
            assert policy == identity, 'Comparison requires the same device, inputs and measurement policy.'
            row = {'trial': trial, 'mode': mode, 'request_seconds': elapsed,
                   'latency_ms': result.latency_ms, 'process': result.record.get('process'),
                   'profile_kernels': result.record['profile']['kernels']}
            rows.append(row)
            print(json.dumps(row), flush=True)
    medians = {mode: float(np.median([row['request_seconds'] for row in rows if row['mode'] == mode]))
               for mode in workers}
    report = {'comparison': policy, 'trials': rows, 'median_request_seconds': medians,
              'request_speedup': medians['fresh'] / medians['reused'],
              'scope': 'same GPU, separate idle endpoints, sequential requests, warmed compiler caches; no kernel speedup claim'}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
