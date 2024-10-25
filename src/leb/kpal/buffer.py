from collections.abc import Buffer
from dataclasses import dataclass, field
from functools import cached_property

from .core.pixel_format import PixelFormat

type ImageShape = tuple[int, int]


@dataclass
class RingBuffer:
    """A circular buffer of contiguous memory for storing raw image data."""

    capacity: int
    shape: ImageShape
    pixel_format: PixelFormat

    buf: memoryview = field(init=False, repr=False)

    _cursor: int = field(init=False, repr=False, default=0)

    def __post_init__(self) -> None:
        num_bits = (
            self.capacity * self.shape[0] * self.shape[1] * self.pixel_format.bits_per_pixel()
        )
        num_bytes, remainder = divmod(num_bits, 8)

        if remainder != 0:
            raise ValueError(
                f"Buffer with requested shape {self.shape} and format {self.pixel_format} would not hold an integer number of bytes"
            )

        self.buf = memoryview(bytearray(num_bytes))

    @property
    def cursor(self) -> int:
        return self._cursor

    @cached_property
    def item_size(self) -> int:
        """Returns the size of a single image in bytes."""
        return self.shape[0] * self.shape[1] * self.pixel_format.bits_per_pixel() // 8

    @cached_property
    def num_bytes(self) -> int:
        """Returns the total number of bytes in the buffer."""
        return self.buf.nbytes

    def put(self, data: Buffer) -> None:
        """Copies data into the RingBuffer."""
        # TODO: Test this!
        loc = (self.cursor * self.item_size) % self.num_bytes

        self.buf[loc : (loc + self.item_size)] = data
