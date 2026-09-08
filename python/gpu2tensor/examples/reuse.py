"""Compare whole-request cost on fresh and reused processes on one device."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from gpu2tensor import Candidate, Evaluator
from gpu2tensor.examples.softmax import sources, trainium_sources, workload


def check_result(result, mode):
    assert result.correct and result.record['profile_status'] == 'ok', result.record
    if result.record['backend'] == 'trainium':
        assert result.record['profile']['summary_status'] == 'ok', result.record
    reused = result.record.get('process', {}).get('mode') == 'reused'
    assert reused == (mode == 'reused'), 'Endpoint has the wrong process mode.'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['cuda', 'trainium'], default='cuda')
    parser.add_argument('--order', choices=['alternating', 'grouped'])
    parser.add_argument('--fresh', required=True)
    parser.add_argument('--reused', required=True)
    parser.add_argument('--trials', type=int, default=4)
    parser.add_argument('--output', type=Path, default=Path('artifacts/reuse'))
    args = parser.parse_args()
    if args.trials < 2:
        parser.error('Use at least two paired trials.')
    order = args.order or ('grouped' if args.backend == 'trainium' else 'alternating')
    if args.backend == 'trainium' and order != 'grouped':
        parser.error('Neuron retains its core in the reused process; run fresh jobs first.')
    workers = {'fresh': Evaluator(args.fresh), 'reused': Evaluator(args.reused)}
    source = sources()[1] if args.backend == 'cuda' else trainium_sources()[1]
    language = 'triton' if args.backend == 'cuda' else 'nki'
    candidate = Candidate(source, language, 'fused_softmax')
    target = workload()
    rows = []

    def warm(mode):
        result = workers[mode].evaluate(candidate, target, profile=True, repetitions=30)
        result.save(args.output / (mode + '-warmup'))
        check_result(result, mode)

    if order == 'alternating':
        for mode in workers:
            warm(mode)
        jobs = [(trial, mode) for trial in range(args.trials)
                for mode in (('fresh', 'reused') if trial % 2 == 0 else ('reused', 'fresh'))]
    else:
        jobs = [(trial, mode) for mode in workers for trial in range(args.trials)]
    policy = None
    for trial, mode in jobs:
        if order == 'grouped' and trial == 0:
            warm(mode)
        started = time.perf_counter()
        result = workers[mode].evaluate(candidate, target, profile=True, repetitions=30)
        elapsed = time.perf_counter() - started
        result.save(args.output / f'{mode}-{trial}')
        check_result(result, mode)
        identity = {key: result.record[key] for key in
                    ('host', 'device', 'inputs', 'timing_method', 'measurement_policy')}
        for key in ('backend', 'torch', 'triton', 'nki', 'neuron_core', 'runtime_neuron_core'):
            if key in result.record:
                identity[key] = result.record[key]
        if policy is None:
            policy = identity
        assert policy == identity, 'Comparison requires the same device, inputs and measurement policy.'
        row = {'trial': trial, 'mode': mode, 'request_seconds': elapsed,
               'latency_ms': result.latency_ms, 'process': result.record.get('process'),
               'profile_kernels': result.record['profile'].get('kernels'),
               'profile_measurements': result.record['profile'].get('measurements')}
        rows.append(row)
        print(json.dumps(row), flush=True)
    medians = {mode: float(np.median([row['request_seconds'] for row in rows if row['mode'] == mode]))
               for mode in workers}
    report = {'comparison': policy, 'order': order, 'trials': rows, 'median_request_seconds': medians,
              'request_speedup': medians['fresh'] / medians['reused'],
              'scope': 'same device, separate idle endpoints, sequential requests after warmup; no kernel speedup claim'}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
