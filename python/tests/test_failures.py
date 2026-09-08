import copy
from gpu2tensor.failures import nki_buffer_diagnostic, nki_compile_diagnostic, classify_saved_nki_failure

MESSAGE = 'error: failed to compile NKI kernel:\nCollected 1 different diagnostics:\n- [x1] error: assertion failed: nc_transpose data must be in [sbuf, psum], got shared_hbm\n'


def test_only_known_frontend_buffer_diagnostics_match():
    assert nki_buffer_diagnostic(MESSAGE)
    assert nki_buffer_diagnostic(MESSAGE.replace('nc_transpose data', 'tensor_copy dst').replace('sbuf, psum','sbuf'))
    for bad in ['internal compiler error', MESSAGE.replace('nc_transpose data','compiler invariant'),MESSAGE.replace('Collected 1','Collected 2'), MESSAGE+'internal compiler error\n']:
        assert nki_buffer_diagnostic(bad) is None


def test_saved_classification_is_separate_and_requires_terminal_frontend_evidence():
    record=dict(backend='trainium',language='nki',status='error',nki='0.6.0',stage='correctness',message=MESSAGE,failure_kind='runtime_or_infrastructure')
    original=copy.deepcopy(record)
    log='compile_kernel_to_nir\nnki/compiler/frontend.py\nAssertionError: '+MESSAGE
    assert classify_saved_nki_failure(record,log)['failure_kind']=='candidate_compile'
    assert record==original
    assert classify_saved_nki_failure(record,'') is None
    assert classify_saved_nki_failure(record,log+'RuntimeError: runtime failed') is None
    assert classify_saved_nki_failure(record | {'status':'timeout'},log) is None


TRANSPOSE = ('error: failed to compile NKI kernel:\nCollected 1 different diagnostics:\n'
             '- [x1] error: assertion failed: Vector engine transpose requires shape <= [32, 32], got [64, 32]\n')


def test_transpose_policy_accepts_only_the_qualified_diagnostic():
    diagnostic = nki_compile_diagnostic(TRANSPOSE)
    assert diagnostic['policy'] == 'nki-0.6-vector-transpose-shape-v1'
    assert nki_compile_diagnostic(MESSAGE) == nki_buffer_diagnostic(MESSAGE)
    for bad in (TRANSPOSE.replace('got [64, 32]', 'got [32, 32]'),
                TRANSPOSE.replace('got [64, 32]', 'got [64, 64]'),
                TRANSPOSE.replace('Collected 1', 'Collected 2'),
                TRANSPOSE + 'internal compiler error\n'):
        assert nki_compile_diagnostic(bad) is None


def test_saved_transpose_failure_requires_frontend_evidence_and_stays_immutable():
    record = dict(backend='trainium', language='nki', status='error', nki='0.6.0',
                  stage='correctness', message=TRANSPOSE, failure_kind='runtime_or_infrastructure')
    original = copy.deepcopy(record)
    log = 'compile_kernel_to_nir\nnki/compiler/frontend.py\nAssertionError: ' + TRANSPOSE
    annotation = classify_saved_nki_failure(record, log)
    assert annotation['diagnostic']['policy'] == 'nki-0.6-vector-transpose-shape-v1'
    assert record == original
    for changes in ({'stage': 'benchmark'}, {'status': 'timeout'}, {'nki': '0.7.0'}, {'latency_ms': [1.0]}):
        assert classify_saved_nki_failure(record | changes, log) is None
    assert classify_saved_nki_failure(record, '') is None
    assert classify_saved_nki_failure(record, log + 'RuntimeError: device fault') is None
