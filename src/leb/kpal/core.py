import asyncio
import importlib
import pkgutil
from dataclasses import dataclass, field
from types import ModuleType
from typing import Optional

import leb.kpal.plugins
from leb.kpal.peripherals import Attribute, Peripheral, Value


class KPALPeripheralError(Exception):
    """Raised when performing an operation on a peripheral"""


class KPALAttributeError(Exception):
    """Raised when performing an operation on a peripheral attribute"""


@dataclass
class Core:
    """The interface between users and peripherals.

    Attributes
    ----------
    peripherals : dict[str, Peripheral]

    """

    peripherals: dict[str, Peripheral] = field(default_factory=dict)
    plugins: dict[str, ModuleType] = field(init=False)

    def __post_init__(self):
        self.plugins = self.import_plugins()

    def import_plugins(self) -> dict[str, ModuleType]:
        # Needed to find any plugins installed after this function was first called
        importlib.invalidate_caches()

        discovered_plugins = {
            name: importlib.import_module(name) for _, name, _ in iter_namespace(leb.kpal.plugins)
        }

        return discovered_plugins

    def resolve_names(self, peripheral_name: str, attribute_name: Optional[str]) -> tuple[Peripheral, Optional[Attribute]]:
        """Resolves peripheral and attribute names into their corresponding objects."""
        if (peripheral := self.peripherals.get(peripheral_name)) is None:
            raise KPALPeripheralError(f"Peripheral {peripheral_name} does not exist")

        if attribute_name is None:
            return peripheral, None

        if (attribute := peripheral.attributes.get(attribute_name)) is None:
            raise KPALAttributeError(
                f"Attribute {attribute_name} does not exist for peripheral {peripheral_name}"
            )

        return peripheral, attribute

    async def build_peripheral(self, peripheral_type: Peripheral, name: str, *args, **kwargs):
        """Creates a new peripheral instance."""
        if name in self.peripherals:
            raise KPALPeripheralError(
                "Cannot create peripheral with name '{name}' because the name already exists"
            )

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
    capacities = {"buf": 4096}
    await core.build_peripheral(plugin, peripheral_name, "hello world", capacities=capacities)
    print(core.peripherals)
    print(core.peripherals[peripheral_name].attributes)

    print(await core.get_attribute(peripheral_name, "foo"))
    print(await core.get_attribute(peripheral_name, "bar"))

    await core.set_attribute(peripheral_name, "bar", 999.0)
    print(await core.get_attribute(peripheral_name, "bar"))


if __name__ == "__main__":
    asyncio.run(main())
