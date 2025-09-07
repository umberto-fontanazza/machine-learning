from numpy import bool, dtype, float64, ndarray, uint8, uint64

type F64Matrix = ndarray[tuple[int, int], dtype[float64]]
type F64Array = ndarray[tuple[int], dtype[float64]]
type BoolArray = ndarray[tuple[int], dtype[bool]]
type U8Array = ndarray[tuple[int], dtype[uint8]]
type U8Matrix = ndarray[tuple[int, int], dtype[uint8]]
type U64Matrix = ndarray[tuple[int, int], dtype[uint64]]
