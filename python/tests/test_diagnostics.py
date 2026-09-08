import numpy as np

from gpu2tensor import Candidate, Evaluator, Workload


def test_input_mutation_is_rejected_even_when_output_is_correct():
    workload = Workload('def reference(x): return x', [(np.ones(4, np.float32),)], read_only_inputs=True)
    candidate = Candidate('def run(x):\n    result = x.clone()\n    x.zero_()\n    return result\n', 'python')
    result = Evaluator(backend='cpu').evaluate(candidate, workload)
    assert result.record['status'] == 'incorrect' and result.latency_ms is None
    assert result.record['diagnostics'][0]['read_only_inputs']['status'] == 'failed'


def test_unsupported_diagnostics_are_explicit():
    workload = Workload('def reference(x): return x', [(np.ones(4, np.float32),)], read_only_inputs=True, poison_outputs=True)
    result = Evaluator(backend='cpu').evaluate(Candidate('def run(x): return x', 'python'), workload, repetitions=1)
    assert result.correct
    assert result.record['diagnostics'][0]['read_only_inputs']['status'] == 'passed'
    assert result.record['diagnostics'][0]['output_poison']['status'] == 'unsupported'
    assert result.record['diagnostics'][0]['bounds']['status'] == 'unsupported'
