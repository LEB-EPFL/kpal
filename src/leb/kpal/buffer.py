"""Shared circular memory buffers backed by NumPy arrays."""

import atexit
import logging
from multiprocessing import shared_memory
from typing import Any, Optional

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


def _bytes_needed(capacity: int, dtype: npt.DTypeLike) -> int:
    """Finds bytes needed to hold an integer number of items within a given capacity."""
    arr_dtype = np.dtype(dtype)
    if capacity < arr_dtype.itemsize:
        raise ValueError(
            "Requested buffer capacity %d is smaller than the size of an item: %s",
            capacity,
            arr_dtype.itemsize,
        )
    num_bytes = (capacity // dtype.itemsize) * dtype.itemsize
    return num_bytes


class BufferedArray(np.ndarray):
    """A shared circular memory buffer backed by a NumPy array.

    The capacity value that is passed to create the buffer is the maximum size of the buffer in
    bytes; the final capacity may be smaller than the requested size to fit an integer number of
    items of type dtype into the buffer.

    """

    def __new__(
        subtype,
        capacity: int,
        name: Optional[str] = None,
        create: bool = False,
        dtype: npt.DTypeLike = np.int16,
        offset: int =0,
        strides: tuple[int, ...] = None,
        order=None,
    ) -> "BufferedArray":
        dtype = np.dtype(dtype)
        capacity = _bytes_needed(capacity, dtype)
        shm = shared_memory.SharedMemory(name=name, create=create, size=capacity)

        size = capacity // dtype.itemsize  # number of items in the buffer
        obj = super().__new__(
            subtype, (size,), dtype=dtype, buffer=shm.buf, offset=offset, strides=strides, order=order
        )

        # Convert from np.ndarray to BufferedArray
        obj = obj.view(subtype)

        obj._writeable = create
        obj._shm = shm
        obj._write_idx = 0
        atexit.register(obj.close)

        # Prevents consumers from modifying the data; don't touch this
        obj.flags.writeable = create

        return obj

    def __array_finalize__(self, obj: None | npt.NDArray[Any], /) -> None:
        if obj is None:
            return

        self._writeable = getattr(obj, "_writeable", False)

    def close(self) -> None:
        self._shm.close()

        if self._writeable:
            try:
                self._shm.unlink()
            except FileNotFoundError:
                logger.debug(
                    "Shared memory buffer %s was already closed and unlinked", self._shm.name
                )

    def put(self, data: npt.ArrayLike):
        """Puts an existing ndarray into the buffer."""
        arr = np.asanyarray(data)

        if arr.nbytes > self.nbytes:
            raise ValueError("Item is too large to put into the buffer")

        if arr.dtype != self.dtype:
            raise ValueError("dtypes of input array and BufferedArrays do not match")

        # TODO Handle wrap-around
        self[self._write_idx:arr.size] = np.ravel(arr)
        self._write_idx += arr.size