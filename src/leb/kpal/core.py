import asyncio
import atexit
import importlib
import logging
import pkgutil
import queue
import threading
from dataclasses import dataclass, field
from types import ModuleType
from typing import Optional

import leb.kpal.plugins
from leb.kpal.peripherals import Attribute, Peripheral, Value


logger = logging.getLogger(__name__)


class KPALPeripheralError(Exception):
    """Raised when performing an operation on a peripheral"""


class Event:
    """A KPAL event to be handled by the event handler"""

class Shutdown(Event):
    """Signals the event handler to shut down"""


def event_handler(event_queue: queue.Queue) -> None:
    logger.debug("Event handler started")
    while True:
        event = event_queue.get()

        match event:
            case Shutdown():
                logger.info("Event handler received the shutdown event. Shutting down...")
                event_queue.task_done()
                break
            case _:
                logger.warning("Not implemented")

    logger.info("Event handler successfully shutdown")


def shutdown_handler(event_queue: queue.Queue, event_thread: threading.Thread) -> None:
    event_queue.put_nowait(Shutdown())
    event_thread.join()


@dataclass
class Core:
    """The interface between users and peripherals.

    Attributes
    ----------
    peripherals : dict[str, Peripheral]
    plugins : dict[str, types.ModuleType]

    """

    peripherals: dict[str, Peripheral] = field(default_factory=dict)
    event_queue: queue.Queue = field(init=False)
    event_thread: threading.Thread = field(init=False)
    plugins: dict[str, ModuleType] = field(init=False)

    def __post_init__(self):
        self.plugins = self.import_plugins()

        self.event_queue = queue.Queue()
        self.event_thread = threading.Thread(
            target=event_handler,
            name="kpal-event-handler",
            args=(self.event_queue,)
        )

        logger.debug("Starting the event handler")
        self.event_thread.start()
        atexit.register(shutdown_handler, self.event_queue, self.event_thread)

    def import_plugins(self) -> dict[str, ModuleType]:
        # Needed to find any plugins installed after this function was first called
        importlib.invalidate_caches()

        discovered_plugins = {
            name: importlib.import_module(name) for _, name, _ in iter_namespace(leb.kpal.plugins)
        }

        return discovered_plugins

    def resolve_names(
        self, peripheral_name: str, attribute_name: Optional[str]
    ) -> tuple[Peripheral, Optional[Attribute]]:
        """Resolves peripheral and attribute names into their corresponding objects."""
        if (peripheral := self.peripherals.get(peripheral_name)) is None:
            raise KPALPeripheralError(f"Peripheral {peripheral_name} does not exist")

        if attribute_name is None:
            return peripheral, None

        if (attribute := peripheral.attributes.get(attribute_name)) is None:
            raise KPALPeripheralError(
                f"Attribute {attribute_name} does not exist for peripheral {peripheral_name}"
            )

        return peripheral, attribute

    async def build_peripheral(self, peripheral_type: Peripheral, name: str, *args, **kwargs):
        """Creates a new peripheral instance."""
        if name in self.peripherals:
            raise KPALPeripheralError(
                "Cannot create peripheral with name '{name}' because the name already exists"
            )

        # The peripheral needs the event queue to send events to the handler
        kwargs.update({"event_queue": self.event_queue})
        peripheral = await peripheral_type.build(*args, **kwargs)

        self.peripherals[name] = peripheral

    async def get_attribute(self, peripheral_name: str, attribute_name: str) -> Value:
        """Gets the value of a peripheral attribute."""
        peripheral, attribute = self.resolve_names(peripheral_name, attribute_name)

        await peripheral.lock.acquire()
        try:
            response = await attribute.getter(peripheral)
        finally:
            peripheral.lock.release()

        return response

    async def set_attribute(self, peripheral_name: str, attribute_name: str, value: Value) -> None:
        """Sets the value of a peripheral attribute."""
        peripheral, attribute = self.resolve_names(peripheral_name, attribute_name)

        await peripheral.lock.acquire()
        try:
            await attribute.setter(peripheral, value)
        finally:
            peripheral.lock.release()


def iter_namespace(ns_pkg: ModuleType):
    # Specifying the second argument (prefix) to iter_modules makes the
    # returned name an absolute name instead of a relative one. This allows
    # import_module to work without having to do additional modification to
    # the name.
    return pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + ".")


async def main():
    core = Core()
    print(core.plugins)

    plugin = core.plugins["leb.kpal.plugins.dummy"].Plugin
    build_args = plugin.build_args()
    print(build_args)

    peripheral_name = "my cool peripheral"
    capacity = 4096
    await core.build_peripheral(plugin, peripheral_name, "hello world", capacity=capacity)
    print(core.peripherals)
    print(core.peripherals[peripheral_name].attributes)

    print(await core.get_attribute(peripheral_name, "foo"))
    print(await core.get_attribute(peripheral_name, "bar"))

    await core.set_attribute(peripheral_name, "bar", 999.0)
    print(await core.get_attribute(peripheral_name, "bar"))

    await asyncio.sleep(0.5)
    core.event_queue.put(Shutdown())


if __name__ == "__main__":
    asyncio.run(main())
