"""Shared circular memory buffers backed by NumPy arrays."""

import atexit
import logging
from multiprocessing import shared_memory
from typing import Any, Optional

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


def _bytes_needed(capacity: int, dtype: npt.DTypeLike) -> int:
    arr_dtype = np.dtype(dtype)
    if capacity < arr_dtype.itemsize:
        raise ValueError(
            "Requested buffer capacity %d is smaller than the size of an item: %s",
            capacity,
            arr_dtype.itemsize,
        )
    num_bytes = (capacity // dtype.itemsize) * dtype.itemsize
    return num_bytes


def _close(buffered_array):
    buffered_array._shm.close()

    if buffered_array._writeable:
        try:
            buffered_array._shm.unlink()
        except FileNotFoundError:
            logger.debug(
                "Shared memory buffer %s was already closed and unlinked", buffered_array._shm.name
            )


class BufferedArray(np.ndarray):
    """A shared circular memory buffer backed by a NumPy array.

    The capacity value that is passed to create the buffer is the maximum size of the buffer in
    bytes; the final capacity may be smaller than the requested size to fit an integer number of
    items of type dtype into the buffer.

    """

    def __new__(
        cls,
        capacity: int,
        name: Optional[str] = None,
        create: bool =False,
        dtype: npt.DTypeLike = np.int16,
        offset: int =0,
        strides: tuple[int, ...] = None,
        order=None,
    ) -> np.ndarray:
        dtype = np.dtype(dtype)
        capacity = _bytes_needed(capacity, dtype)
        shm = shared_memory.SharedMemory(name=name, create=create, size=capacity)

        size = capacity // dtype.itemsize  # number of items in the buffer
        obj = super().__new__(
            cls, (size,), dtype=dtype, buffer=shm.buf, offset=offset, strides=strides, order=order
        )

        obj._writeable = create
        obj._shm = shm
        atexit.register(_close, obj)

        # Prevents consumers from modifying the data; don't touch this
        obj.flags.writeable = create

        return obj

    def __array_finalize__(self, obj: None | npt.NDArray[Any], /) -> None:
        if obj is None:
            return

        self._writeable = getattr(obj, "_writeable", False)
        self._shm = getattr(obj, "_shm")