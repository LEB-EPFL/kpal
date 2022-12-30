"""Peripherals provide an interface between hardware and the control system software."""

import asyncio
import inspect
import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Protocol, TypeAlias

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
    """A peripheral attribute and its associated getter and setter."""

    description: str
    getter: Callable[[], Value]
    setter: Callable[[Value], None]


class Peripheral(Protocol):
    """The interface to a hardware device.
    
    All peripherals have a lock that is used by the core to serialize access to the device.

    """

    attributes: dict[str, Attribute]
    lock: asyncio.Lock
    _state: PeripheralState

    def __init__(self) -> None:
        """Any actual hardware initialization should be done in the build method."""
        self.lock = asyncio.Lock()
        self._state = PeripheralState.PREINIT

    @classmethod
    def build_args(cls) -> dict[str, inspect.Parameter]:
        """Returns the set of arguments that are required to build the peripheral.

        This method traverses the class hierarchy for the peripheral instance and discovers the
        arguments of the build methods of all the parent classes. Classes are traversed in the
        order of the Python MRO: left to right, then bottom to top.

        """
        build_args = {}

        # Skip the first class in the MRO, which is the class itself.
        for klass in cls.__mro__:
            try:
                if klass.__name__ == "Peripheral":
                    logger.debug(
                        "Stopping build argument discovery of class %s because Peripheral class "
                        "was reached while traversing its MRO",
                        cls,
                    )
                    break

                # Filter out self, args, kwargs, and _ arguments
                parameters = {
                    k: v
                    for k, v in inspect.signature(klass.build).parameters.items()  # type: ignore
                    if k not in ["self", "args", "kwargs", "_"]
                }

                if duplicates := [k for k in parameters.keys() if k in build_args]:
                    logger.warning(
                        "Duplicate build_args detected for parent %s of class %s: %s. Two "
                        "different Mixins are using the same build method argument names",
                        klass,
                        cls,
                        duplicates,
                    )

                build_args.update(parameters)

            except AttributeError:
                logger.debug(
                    "Could not find build method of parent class %s while discovering build kwargs "
                    "for class %s",
                    klass,
                    cls,
                )
                continue

        return build_args

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

    term_chars: bytes = field(init=False)

    _reader: asyncio.StreamReader = field(init=False, repr=False)
    _writer: asyncio.StreamWriter = field(init=False, repr=False)

    def __post_init__(self) -> None:
        super().__init__()

    async def build(self, url: str, baudrate: int = 115200, term_chars: bytes = b"\n", **_):
        """Initializes the serial mixin data and opens a serial connection to the device."""
        self.term_chars = term_chars
        self._reader, self._writer = await serial_asyncio.open_serial_connection(
            url=url, baudrate=baudrate
        )

    async def recv(self) -> None:
        """Receives a message from the device."""
        _ = await self._reader.readuntil(self.term_chars)

    async def send(self, msg: bytes) -> None:
        """Sends a message to the device."""
        self._writer.write(msg)
