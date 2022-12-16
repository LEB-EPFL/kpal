"""Peripherals provide an interface between hardware and the control system software."""

from asyncio import StreamReader, StreamWriter
from dataclasses import InitVar, dataclass, field
from enum import IntEnum
import inspect
import logging
from typing import Callable, Protocol, TypeAlias
from uuid import UUID

import serial_asyncio


logger = logging.getLogger(__name__)


class PeripheralState(IntEnum):
    """Peripherals must be in exactly one of these states at any given time."""

    PREINIT = 0
    INIT = 1
    RUNNING = 2
    SHUTDOWN = 3
    POSTSHUTDOWN = 4
    ERROR = -99


Value: TypeAlias = bytes | float | int | str


@dataclass(frozen=True)
class Attribute:
    name: str
    getter: Callable[[], Value]
    setter: Callable[[Value], None]


class Peripheral(Protocol):
    """The interface to a hardware device."""

    attributes: dict[UUID, Attribute]
    _state: PeripheralState

    def __init__(self) -> None:
        """Any actual hardware initialization should be done in the"""
        self._state = PeripheralState.PREINIT

    def attributes(self) -> dict[UUID, Attribute]:
        """Returns the set of the peripherial's attributes."""

    @classmethod
    def build_args(cls):
        """Returns the set of keyword arguments that are required to build the peripheral.
        
        This method traverses the class hierarchy for the peripheral instance and discovers the
        kwargs of the build methods of all the parent classes. Classes are traversed in the
        order of the Python MRO: left to right, then bottom to top.

        """
        # Skip the first class in the MRO, which is the class itself.
        for klass in cls.__mro__[1:]:
            try:
                if klass.__name__ == "Peripheral":
                    logger.debug(
                        "Stopping build argument discovery of class %s because Peripheral class "
                        "was reached while traversing its MRO",
                        cls
                    )
                    break

                print(klass, inspect.signature(klass.build))
            except AttributeError:
                logger.debug(
                    "Could not find build method of parent class %s while discovering build kwargs "
                    "for class %s",
                    klass,
                    cls
                )

    @classmethod
    async def build(cls, *args, **kwargs) -> "Peripheral":
        """Builds an instance of the peripheral, performing any hardware initialization."""


    @property
    def state(self) -> PeripheralState:
        """Returns the state of the peripheral."""
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

    async def build(self, url: str, baudrate: int = 115200, term_chars: bytes = b"\n", **kwargs):
        """Initializes the serial mixin data and opens a serial connection to the device."""
        self._term_chars = term_chars
        self._reader, self._writer = await serial_asyncio.open_serial_connection(
            url=url, baudrate=baudrate
        )

    async def recv(self) -> None:
        """Receives a message from the device."""
        _ = await self._reader.readuntil(self._term_chars)

    async def send(self, msg: bytes) -> None:
        """Sends a message to the device."""
        self._writer.send(msg)
