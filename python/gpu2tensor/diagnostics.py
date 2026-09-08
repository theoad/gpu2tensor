"""Small empirical buffer checks; these do not prove memory safety."""

import hashlib

from gpu2tensor.tensors import input_storage


def fingerprint(value):
    storage, dtype = input_storage(value)
    return dtype, tuple(storage.shape), hashlib.sha256(storage.tobytes()).hexdigest()


def report(*, read_only_inputs, poison_outputs, inputs_unchanged=None, outputs_match=None):
    return {
        "read_only_inputs": {"status": "not_requested" if not read_only_inputs else
                             "unsupported" if inputs_unchanged is None else "passed" if inputs_unchanged else "failed"},
        "output_poison": {"status": "not_requested" if not poison_outputs else
                          "unsupported" if outputs_match is None else "passed" if outputs_match else "failed"},
        "bounds": {"status": "unsupported"}, "lifetime": {"status": "unsupported"},
        "race_detection": {"status": "unsupported"},
    }
