import pytest

from leb.kpal.peripherals import Peripheral, SerialMixin


class SerialPeripheral(SerialMixin, Peripheral):
    @classmethod
    def build(cls, *args, **kwargs) -> "SerialPeripheral":
        peripheral = cls()
        url, = args
        SerialMixin.build(peripheral, url, **kwargs)


@pytest.fixture
def serial_peripheral():
    return SerialPeripheral
