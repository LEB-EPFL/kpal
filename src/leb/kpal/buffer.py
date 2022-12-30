"""Shared memory buffers backed by NumPy arrays."""

import atexit
import logging
from dataclasses import InitVar, dataclass, field
from multiprocessing import shared_memory
from typing import Optional

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


def _bytes_needed(capacity: int, dtype: npt.DTypeLike) -> int:
    arr_dtype = np.dtype(dtype)
    if capacity < arr_dtype.itemsize:
        raise ValueError(
            "Requested buffer capacity %d is smaller than the size of an item: %s",
            capacity,
            arr_dtype.itemsize
        )
    num_bytes = (capacity // dtype.itemsize) * dtype.itemsize
    return num_bytes


@dataclass
class Buffer:
    """A shared memory buffer backed by a NumPy array.
    
    The capacity value that is passed to create the buffer is the maximum size of the buffer in
    bytes; the final capacity may be smaller than the requested size to fit an integer number of
    items of type dtype into the buffer.

    """

    capacity: InitVar[int]  # bytes
    dtype: InitVar[npt.DTypeLike] = np.int16
    name: InitVar[Optional[str]] = None
    create: InitVar[bool] = False

    data: npt.NDArray = field(init=False)

    _shm: shared_memory.SharedMemory = field(init=False, repr=False)
    _writeable: bool = field(init=False, repr=False)

    def __post_init__(self, capacity: int, dtype: npt.DTypeLike, name: Optional[str], create: bool):
        self._writeable = create

        dtype = np.dtype(dtype)
        capacity = _bytes_needed(capacity, dtype)
        self._shm = shared_memory.SharedMemory(name=name, create=create, size=capacity)
        atexit.register(self.close)
        
        size = capacity // dtype.itemsize  # number of items in the buffer
        self.data = np.ndarray((size,), dtype=dtype, buffer=self._shm.buf)

        # Prevents consumers from modifying the data; don't touch this
        self.data.flags.writeable = self._writeable

    def close(self):
        self._shm.close()

        if self._writeable:
            try:
                self._shm.unlink()
            except FileNotFoundError:
                logger.debug(
                    "Shared memory buffer %s was already closed and unlinked", self._shm.name
                )
