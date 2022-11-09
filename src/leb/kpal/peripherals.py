"""Peripherals are the interface between hardware and the control system."""

import inspect
from typing import Any


def get_attributes(peripheral: "Peripheral") -> list[Any]:
    """Returns a list of all of a peripheral's attributes."""
    built_in_attributes = dir(type("dummy", (object,), {}))
    return [item for item in inspect.getmembers(peripheral) if item[0] not in built_in_attributes]


class Peripheral:
    """The interface to a hardware device."""


class SerialPeripheral(Peripheral):
    """A hardware device that employs serial communication."""

    pass
