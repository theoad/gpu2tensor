"""Logical tensor dtypes and exact wire storage, with lazy Torch imports."""

import numpy as np


def logical_dtype(value):
    if isinstance(value, np.ndarray):
        return value.dtype.name
    return str(value.dtype).removeprefix("torch.")


def input_storage(value):
    """Return a contiguous NumPy view and its logical dtype without rounding."""
    if isinstance(value, np.ndarray):
        if not value.flags.c_contiguous:
            raise ValueError("The input contract requires C-contiguous arrays.")
        if value.dtype.name == "bfloat16":
            return value.view(np.uint16), "bfloat16"
        if value.dtype.kind not in "biuf":
            raise TypeError("Inputs must be real numeric arrays or CPU Torch tensors.")
        return value, value.dtype.name
    if not hasattr(value, "detach"):
        raise TypeError("Inputs must be real numeric arrays or CPU Torch tensors.")
    import torch
    if not isinstance(value, torch.Tensor) or value.is_complex():
        raise TypeError("Inputs must be real numeric arrays or CPU Torch tensors.")
    if value.device.type != "cpu":
        raise ValueError("Workload inputs must be on CPU; call .cpu() explicitly.")
    if not value.is_contiguous():
        raise ValueError("The input contract requires contiguous tensors.")
    value = value.detach()
    if value.dtype == torch.bfloat16:
        return value.view(torch.int16).numpy().view(np.uint16), "bfloat16"
    return value.numpy(), logical_dtype(value)


def decode_storage(storage, dtype):
    if dtype == "bfloat16":
        if storage.dtype.kind != "u" or storage.dtype.itemsize != 2:
            raise ValueError("BF16 wire storage must use uint16 bits.")
        import torch
        native = storage.astype(np.uint16, copy=False)
        return torch.from_numpy(native.view(np.int16)).view(torch.bfloat16)
    if storage.dtype.name != dtype:
        raise ValueError("Logical dtype differs from tensor storage.")
    return storage


def copy_input(value):
    return value.copy() if isinstance(value, np.ndarray) else value.detach().clone()


def torch_input(value):
    import torch
    if isinstance(value, np.ndarray):
        if value.dtype.name == "bfloat16":
            return torch.from_numpy(value.view(np.int16)).view(torch.bfloat16)
        return torch.from_numpy(value)
    return value.detach()


def numpy_input(value):
    """NKI receives a native NumPy BF16 dtype; this SDK-side extra is optional."""
    if isinstance(value, np.ndarray):
        return value
    storage, dtype = input_storage(value)
    if dtype == "bfloat16":
        import ml_dtypes
        return storage.view(ml_dtypes.bfloat16)
    return storage


def host_array(value):
    """Values for comparison; check logical_dtype before calling this helper."""
    if isinstance(value, np.ndarray):
        return value.astype(np.float32) if value.dtype.name == "bfloat16" else value
    value = value.detach().cpu()
    if logical_dtype(value) == "bfloat16":
        value = value.float()
    return value.numpy()
