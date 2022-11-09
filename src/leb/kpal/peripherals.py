"""Peripherals provide an interface between hardware and the control system software."""

from asyncio import StreamReader, StreamWriter
from dataclasses import InitVar, dataclass, field
import inspect
from typing import Any

import serial_asyncio


def get_attributes(peripheral: "Peripheral") -> list[Any]:
    """Returns a list of all of a peripheral's attributes."""
    built_in_attributes = dir(type("dummy", (object,), {}))
    return [item for item in inspect.getmembers(peripheral) if item[0] not in built_in_attributes]


class Peripheral:
    """The interface to a hardware device."""


@dataclass
class SerialPeripheral(Peripheral):
    term_chars: InitVar[bytes] = "\n"

    _reader: StreamReader = field(init=False, repr=False)
    _writer: StreamWriter = field(init=False, repr=False)

    """A hardware device that employs serial communication."""
    def __post_init__(self, term_chars: bytes) -> None:
        self._term_chars = term_chars

    @classmethod
    async def create(cls, url: str, baudrate:int = 115200):
        """Creates a new serial peripheral and opens a serial connection to the device.
        
        """
        self = SerialPeripheral()
        self._reader, self._writer = await serial_asyncio.open_serial_connection(
            url,
            baudrate=baudrate
        )
        return self

    async def rx(self) -> None:
        """Receives a message from the device."""
        _ = self._reader.readuntil(self._term_chars)

    async def tx(self, msg: bytes) -> None:
        """Transmits a message to the device."""
        self._writer.send(msg)