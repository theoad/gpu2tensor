import copy
from gpu2tensor.failures import nki_buffer_diagnostic, classify_saved_nki_failure

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
