import numpy as np
from gpu2tensor import Candidate, Workload
from gpu2tensor.client import pack_request, unpack_result
from gpu2tensor.process import Process
from gpu2tensor.runner import evaluate_archive


def evaluate(process, source, timeout=30):
    target=Workload('def reference(x): return x', [(np.ones(4, np.float32),)])
    payload=pack_request(Candidate(source,'python'),target,False,1)
    return unpack_result(evaluate_archive(payload,'cpu',timeout,process=process))


def test_reuses_imports_but_restores_precision_flags_and_recycles():
    process=Process(max_requests=3)
    try:
        source='import torch\ndef run(x):\n    torch.backends.cuda.matmul.allow_tf32 = True\n    return x\n'
        first=evaluate(process,source)
        second=evaluate(process,'import torch\ndef run(x):\n    assert not torch.backends.cuda.matmul.allow_tf32\n    return x\n')
        third=evaluate(process,'def run(x): return x')
        fourth=evaluate(process,'def run(x): return x')
        assert all(r.correct for r in [first,second,third,fourth])
        assert first.record['process']['pid']==second.record['process']['pid']==third.record['process']['pid']
        assert fourth.record['process']['pid'] != first.record['process']['pid']
    finally:
        process.close()


def test_timeout_and_exit_recycle_the_child():
    process=Process()
    try:
        first=evaluate(process,'def run(x): return x')
        timeout=evaluate(process,'import time\ndef run(x):\n    time.sleep(60)\n    return x\n',timeout=0.1)
        assert timeout.record['status']=='timeout'
        recovered=evaluate(process,'def run(x): return x')
        assert recovered.correct and recovered.record['process']['pid'] != first.record['process']['pid']
        failed=evaluate(process,'import os\nos._exit(7)\ndef run(x): return x')
        assert failed.record['status']=='worker_error'
        assert evaluate(process,'def run(x): return x').correct
    finally:
        process.close()
