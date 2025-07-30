from numpy import bool, dtype, float64, int8, ndarray

type FloatMatrix = ndarray[tuple[int, int], dtype[float64]]
type IntArray = ndarray[tuple[int], dtype[int8]]
type BoolArray = ndarray[tuple[int], dtype[bool]]
