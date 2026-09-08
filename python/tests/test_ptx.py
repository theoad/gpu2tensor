import io
import json
import zipfile

import numpy as np
import pytest

from gpu2tensor import Candidate, Workload
from gpu2tensor.client import pack_request
from gpu2tensor.ptx import Input, Launch, Output, Scalar, Workspace


def launch(**changes):
    fields = dict(entry='matmul', grid=(8, 1, 1), block=(128, 1, 1),
                  arguments=(Input(0, alignment=2), Input(1, alignment=2), Output((32, 32), 'bfloat16', alignment=2),
                             Scalar('uint32', 32), Scalar('uint32', 32), Scalar('uint32', 32),
                             Workspace((128,), 'uint8')))
    return Launch(**(fields | changes))


def test_explicit_ptx_contract_roundtrips_without_cuda():
    plan = launch(shared_memory_bytes=128)
    candidate = Candidate('.visible .entry matmul() {}', 'ptx', launch=plan)
    workload = Workload('def reference(x): return x', [(np.ones(3, np.float32),)])
    with zipfile.ZipFile(io.BytesIO(pack_request(candidate, workload, False, 1))) as archive:
        request = json.loads(archive.read('request.json'))
    assert Launch.from_dict(request['launch']) == plan
    assert request['language'] == 'ptx'


@pytest.mark.parametrize('changes', [dict(block=(1025, 1, 1)), dict(grid=(0, 1, 1)),
    dict(arguments=(Input(0),)), dict(arguments=(Output((3,), 'bfloat16'), Scalar('uint32', -1))),
    dict(arguments=(Output((3,), 'bfloat16', alignment=3),)), dict(shared_memory_bytes=-1)])
def test_invalid_launches_are_rejected(changes):
    with pytest.raises(ValueError):
        launch(**changes)


def test_ptx_requires_launch():
    with pytest.raises(ValueError, match='Launch'):
        Candidate('ptx', 'ptx')
