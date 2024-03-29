import queue

import numpy as np

from leb.kpal.peripherals import Attribute, Peripheral, PeripheralState, ProducerMixin


async def get_foo(self: "Plugin") -> int:
    return self.foo


async def set_foo(self: "Plugin", value: int) -> None:
    """Sets the value of foo"""
    self.foo = value


async def get_bar(self: "Plugin") -> float:
    return self.bar


async def set_bar(self: "Plugin", value: float) -> None:
    """Sets the value of bar"""
    self.bar = value


async def get_baz(self: "Plugin") -> str:
    return self.baz


async def set_baz(self: "Plugin", value: str) -> None:
    """Sets the value of baz"""
    self.baz = value


class Plugin(ProducerMixin, Peripheral):
    attributes = {
        "foo": Attribute(
            description="foo",
            getter=get_foo,
            setter=set_foo,
        ),
        "bar": Attribute(
            description="bar",
            getter=get_bar,
            setter=set_bar,
        ),
        "baz": Attribute(
            description="baz",
            getter=get_baz,
            setter=set_baz,
        ),
    }

    def __init__(self, event_queue: queue.Queue) -> None:
        super().__init__(event_queue)
        self.foo = 42
        self.bar = 42.0
        self.baz = "42"

    @classmethod
    async def build(cls, msg: str, **kwargs) -> "Plugin":
        peripheral = cls(kwargs["event_queue"])
        peripheral._state = PeripheralState.INIT

        await ProducerMixin.build(peripheral, kwargs["capacity"])

        peripheral._state = PeripheralState.RUNNING
        print(msg)

        return peripheral

    async def produce(self):
        dtype = self.buffer.dtype
        data = np.array([0, 1], dtype=dtype)
        
        self.buffer.put(data)
