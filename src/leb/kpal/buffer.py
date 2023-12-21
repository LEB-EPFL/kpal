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
        offset: int = 0,
        strides: tuple[int, ...] = None,
        order=None,
    ) -> "BufferedArray":
        dtype = np.dtype(dtype)
        capacity = _bytes_needed(capacity, dtype)
        shm = shared_memory.SharedMemory(name=name, create=create, size=capacity)

        size = capacity // dtype.itemsize  # maximum number of items in the buffer
        obj = super().__new__(
            subtype,
            (size,),
            dtype=dtype,
            buffer=shm.buf,
            offset=offset,
            strides=strides,
            order=order,
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

    def put(self, data: npt.NDArray):
        """Puts an existing ndarray into the buffer at the next available location."""
        self._validate(data)

        # If self.size = 8, self._write_idx = 5, and arr.size = 4, then current_write_idx is -3 and
        # next_write_idx is 1.
        current_write_idx = self._write_idx
        quotient, mod = divmod(self._write_idx + data.size, self.size)
        if quotient > 0:
            # Array will be inserted partly at the end and partly at the beginning of the buffer
            current_write_idx = self._write_idx - self.size
            next_write_idx = mod

            assert current_write_idx < 0
        else:
            next_write_idx = self._write_idx + data.size

        if next_write_idx > self.shape[0]:
            # Integer ranges won't work when wrapping around the end of the array, so use an array of
            # indexes instead
            indexes = list(range(current_write_idx, next_write_idx))
            self[indexes] = np.ravel(data)
        else:
            self[current_write_idx:next_write_idx] = np.ravel(data)

        self._write_idx = next_write_idx

    def _validate(self, arr: np.ndarray):
        """Validates input array data."""
        if arr.nbytes > self.nbytes:
            raise ValueError("Item is too large to put into the buffer")

        if arr.dtype != self.dtype:
            raise ValueError("dtypes of input array and BufferedArrays do not match")


if __name__ == "__main__":
    capacity = 16  # bytes
    dtype = np.int16  # 2 bytes / integer

    arr = BufferedArray(capacity=capacity, create=True, dtype=dtype)

    item1 = np.array([1, 1, 1, 1, 1], dtype)
    item2 = np.array([2, 2, 2, 2], dtype)
    item3 = np.array([3], dtype)

    arr.put(item1)
    assert np.all(arr[0:5] == 1)
    assert np.all(arr[5:] == 0)

    arr.put(item2)
    assert arr[0] == 2
    assert np.all(arr[1:5] == 1)
    assert np.all(arr[5:] == 2)

    arr.put(item3)
    assert arr[0] == 2
    assert arr[1] == 3
    assert np.all(arr[2:5] == 1)
    assert np.all(arr[5:] == 2)

    arr.close()
