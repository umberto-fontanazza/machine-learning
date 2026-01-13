from numpy import dtype, float64, int8, ndarray, uint8

type F64Matrix = ndarray[tuple[int, int], dtype[float64]]
type U8Matrix = ndarray[tuple[int, int], dtype[uint8]]
type F64Array = ndarray[tuple[int], dtype[float64]]
type U8Array = ndarray[tuple[int], dtype[uint8]]
type I8Array = ndarray[tuple[int], dtype[int8]]
