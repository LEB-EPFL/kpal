import numpy as np

from .raw_image import RawImage, RawImageShape

type Index = int


class RingBuffer(np.ndarray):
    """A contiguous, circular array of raw image data."""

    def __new__(
        subtype,
        capacity: int,
        shape: RawImageShape,
        dtype=float,
    ):
        """Creates a new RingBuffer.

        Parameters
        ----------
        capacity : int
            The maximum number of images that the buffer can hold.
        shape : ImageShape
            The shape of the images in the buffer.
        dtype : np.dtype
            The data type of the buffer.

        """
        dtype = np.dtype(dtype)
        final_shape = (capacity, *shape)
        obj = super().__new__(subtype, final_shape, dtype, None, 0, None, None)

        # Convert from np.ndarray to RingBuffer
        obj = obj.view(subtype)

        obj._write_idx = 0

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

    def put(self, image: RawImage) -> Index:
        """Copies an existing Image into the buffer at the next available location.

        If the buffer is full, the oldest image will be overwritten.

        Parameters
        ----------
        image : Image
            The image to insert into the buffer.

        Returns
        -------
        Index
            The index of the inserted image.

        """
        self[self._write_idx] = image.data
        self._write_idx: int = (self._write_idx + 1) % len(self)

        return self._write_idx
