import numpy as np


def rand_numpy(shape, dtype, **kwargs):
    if dtype in (np.float16, np.float32, np.float64):
        return np.random.rand(*shape).astype(dtype)
    if dtype in (np.int16, np.int32, np.int64):
        return np.random.randint(size=shape, dtype=dtype, **kwargs)
    raise RuntimeError(f"got unexpected dtype '{dtype}'")
