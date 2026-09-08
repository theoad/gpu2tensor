"""Explicit positional arguments for a complete PTX entry point."""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class Input:
    index: int
    alignment: int = 1


@dataclass(frozen=True)
class Output:
    shape: tuple[int, ...]
    dtype: str
    alignment: int = 1


@dataclass(frozen=True)
class Workspace:
    shape: tuple[int, ...]
    dtype: str = "uint8"
    initialization: str = "zero"
    alignment: int = 1


@dataclass(frozen=True)
class Scalar:
    dtype: str
    value: int | float


SCALAR_TYPES = {"int32", "uint32", "int64", "uint64", "float32", "float64"}
BUFFER_TYPES = SCALAR_TYPES | {"float16", "bfloat16", "int8", "uint8", "int16", "uint16"}
ARGUMENT_TYPES = {cls.__name__.lower(): cls for cls in (Input, Output, Workspace, Scalar)}


@dataclass(frozen=True)
class Launch:
    entry: str
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    arguments: tuple[Input | Output | Workspace | Scalar, ...]
    shared_memory_bytes: int = 0

    def __post_init__(self):
        if not self.entry or not self.entry.isascii() or '\x00' in self.entry:
            raise ValueError("Provide an ASCII entry point name.")
        for dimensions in (self.grid, self.block):
            if len(dimensions) != 3 or any(type(v) is not int or not 1 <= v <= 2**32 - 1 for v in dimensions):
                raise ValueError("Grid and block each require three positive uint32 dimensions.")
        if math.prod(self.block) > 1024:
            raise ValueError("A CUDA block cannot exceed 1024 threads.")
        if type(self.shared_memory_bytes) is not int or not 0 <= self.shared_memory_bytes <= 2**32 - 1:
            raise ValueError("Shared memory size must be a nonnegative uint32.")
        if sum(isinstance(a, Output) for a in self.arguments) != 1:
            raise ValueError("Declare exactly one output buffer.")
        for argument in self.arguments:
            if type(argument) not in ARGUMENT_TYPES.values():
                raise TypeError("Unknown PTX argument type.")
            if isinstance(argument, Scalar):
                if argument.dtype not in SCALAR_TYPES:
                    raise ValueError("Unsupported scalar dtype.")
                if argument.dtype.startswith(("int", "uint")):
                    signed = argument.dtype.startswith("int")
                    bits = int(argument.dtype.removeprefix("u").removeprefix("int"))
                    lower, upper = (-2**(bits-1), 2**(bits-1)) if signed else (0, 2**bits)
                    if type(argument.value) is not int or not lower <= argument.value < upper:
                        raise ValueError("Scalar value is outside its declared integer range.")
                elif not math.isfinite(argument.value):
                    raise ValueError("Scalar values must be finite.")
            else:
                if type(argument.alignment) is not int or argument.alignment < 1 or argument.alignment & (argument.alignment - 1):
                    raise ValueError("Buffer alignment must be a positive power of two.")
                if isinstance(argument, Input):
                    if type(argument.index) is not int or argument.index < 0:
                        raise ValueError("Input index must be nonnegative.")
                else:
                    if argument.dtype not in BUFFER_TYPES or not argument.shape or any(type(v) is not int or v < 1 for v in argument.shape):
                        raise ValueError("Buffer shape and dtype must describe a nonempty numeric tensor.")
                    if isinstance(argument, Workspace) and argument.initialization not in ("zero", "uninitialized"):
                        raise ValueError("Workspace initialization must be zero or uninitialized.")

    def to_dict(self):
        result = asdict(self)
        result["arguments"] = [{"kind": type(a).__name__.lower(), **asdict(a)} for a in self.arguments]
        return result

    @classmethod
    def from_dict(cls, value):
        arguments = []
        for item in value["arguments"]:
            item = dict(item)
            kind = item.pop("kind")
            if "shape" in item:
                item["shape"] = tuple(item["shape"])
            arguments.append(ARGUMENT_TYPES[kind](**item))
        return cls(value["entry"], tuple(value["grid"]), tuple(value["block"]), tuple(arguments),
                   value.get("shared_memory_bytes", 0))
