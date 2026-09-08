import torch

from gpu2tensor import Evaluator, Workload
from gpu2tensor.vendor import matmul


def test_vendor_factory_keeps_source_and_language_explicit():
    for backend in ('cuda', 'trainium'):
        candidate = matmul(backend)
        assert candidate.language == 'torch' and candidate.name == 'vendor_matmul'
        assert 'torch.matmul(a, b)' in candidate.source


def test_vendor_source_cpu_plumbing_uses_same_validator_contract():
    inputs = torch.eye(4, dtype=torch.bfloat16)
    target = Workload('def reference(a, b): return (a.double() @ b.double()).bfloat16()', [(inputs, inputs)],
                      validator="def validate(inputs, actual): return {'passed': bool((actual == inputs[0]).all())}",
                      capture_outputs=True)
    result = Evaluator(backend='cpu').evaluate(matmul('trainium'), target, repetitions=1)
    assert result.correct and result.record['outputs'][0]['dtype'] == 'bfloat16'


def test_default_and_strict_baselines_are_distinct():
    default = matmul('cuda')
    strict = matmul('cuda', precision='strict')
    assert 'allow_bf16_reduced_precision_reduction' not in default.source
    assert 'allow_bf16_reduced_precision_reduction = False' in strict.source
    assert default.name != strict.name
