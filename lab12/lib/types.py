from typing import Callable, Sequence

from numpy import bool, dtype, float64, int8, ndarray, uint8, uint64

type F64Matrix = ndarray[tuple[int, int], dtype[float64]]
type F64Array = ndarray[tuple[int], dtype[float64]]
type BoolArray = ndarray[tuple[int], dtype[bool]]
type U8Array = ndarray[tuple[int], dtype[uint8]]
type U64Matrix = ndarray[tuple[int, int], dtype[uint64]]
type I8Array = ndarray[tuple[int], dtype[int8]]
type Kernel = Callable[[F64Matrix, F64Matrix], F64Matrix]
type F64Tensor3D = ndarray[tuple[int, int, int], dtype[float64]]
type GmmComponent = tuple[float, F64Array, F64Matrix]
type Gmm = Sequence[GmmComponent]
