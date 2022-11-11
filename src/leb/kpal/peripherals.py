"""Peripherals provide an interface between hardware and the control system software."""

from asyncio import StreamReader, StreamWriter
from dataclasses import InitVar, dataclass, field
from enum import Enum, auto
from typing import Callable, Protocol, TypeAlias

import serial_asyncio


class PeripheralState(Enum):
    """Peripherals must be in exactly one of these states at any given time."""

    PREINIT = auto()
    INIT = auto()
    RUNNING = auto()
    SHUTDOWN = auto()
    POSTSHUTDOWN = auto()
    ERROR = auto()


ValueTypes: TypeAlias = bytes | float | int | str

Value: TypeAlias = ValueTypes | Callable[..., ValueTypes]

Attributes: TypeAlias = dict[tuple[str] | tuple[str, PeripheralState], Value]


class Peripheral(Protocol):
    """The interface to a hardware device."""

    attributes: Attributes

    @property
    def state(self) -> PeripheralState:
        return self._state


@dataclass
class SerialMixin:
    """A hardware device that employs serial communication."""

    term_chars: InitVar[bytes] = b"\n"

    _reader: StreamReader = field(init=False, repr=False)
    _writer: StreamWriter = field(init=False, repr=False)

    def __post_init__(self, term_chars: bytes) -> None:
        super().__init__()
        self._term_chars = term_chars

    @classmethod
    async def create(cls, url: str, baudrate: int = 115200, term_chars: bytes = b"\n"):
        """Creates a new serial peripheral and opens a serial connection to the device."""
        self = cls(term_chars)
        self._reader, self._writer = await serial_asyncio.open_serial_connection(
            url=url, baudrate=baudrate
        )
        return self

    async def recv(self) -> None:
        """Receives a message from the device."""
        _ = await self._reader.readuntil(self._term_chars)

    async def send(self, msg: bytes) -> None:
        """Sends a message to the device."""
        self._writer.send(msg)
