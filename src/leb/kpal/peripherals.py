"""Peripherals provide an interface between hardware and the control system software."""

from asyncio import StreamReader, StreamWriter
from dataclasses import InitVar, dataclass, field
from enum import IntEnum, auto
from typing import Callable, Protocol, TypeAlias

import serial_asyncio


class PeripheralState(IntEnum):
    """Peripherals must be in exactly one of these states at any given time."""

    PREINIT = 0
    INIT = 1
    RUNNING = 2
    SHUTDOWN = 3
    POSTSHUTDOWN = 4
    ERROR = -1


ValueTypes: TypeAlias = bytes | float | int | str

Value: TypeAlias = ValueTypes | Callable[..., ValueTypes]

Attributes: TypeAlias = dict[tuple[str] | tuple[str, PeripheralState], Value]


class Peripheral(Protocol):
    """The interface to a hardware device."""

    attributes: Attributes
    _state: PeripheralState

    def __init__(self) -> None:
        self._state = PeripheralState.PREINIT

    @property
    def state(self) -> PeripheralState:
        return self._state
    
    @state.setter
    def state(self, value: PeripheralState) -> None:
        if value == PeripheralState.ERROR:
            self._state = value
            return

        if self._state != value - 1:
            raise ValueError(f"Cannot advance peripheral from {self._state} to {value}")

        self._state = value


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
