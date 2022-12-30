"""Shared memory buffers backed by NumPy arrays."""

import atexit
import logging
from dataclasses import InitVar, dataclass, field
from multiprocessing import shared_memory
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Buffer:
    """A shared memory buffer backed by a NumPy array."""

    capacity: int  # bytes
    name: InitVar[Optional[str]] = None
    create: InitVar[bool] = False

    _shm: shared_memory.SharedMemory = field(init=False, repr=False)
    _writeable: bool = field(init=False, repr=False)

    def __post_init__(self, name: Optional[str], create: bool):
        self._writeable = create
        self._shm = shared_memory.SharedMemory(name=name, create=create, size=self.capacity)

        atexit.register(self.close)

    def close(self):
        self._shm.close()

        if self._writeable:
            try:
                self._shm.unlink()
            except FileNotFoundError:
                logger.debug(
                    "Shared memory buffer %s was already closed and unlinked", self._shm.name
                )
