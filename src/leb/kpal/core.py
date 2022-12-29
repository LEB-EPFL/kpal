from dataclasses import dataclass, field
from typing import Optional

from leb.kpal.peripherals import Peripheral, Value


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

    def resolve_names(self, peripheral_name: str, attribute_name: Optional[str]) -> tuple(str, str):
        """Resolves peripheral and attribute names into their corresponding objects."""
        if peripheral := self.peripherals.get(peripheral_name) is None:
            raise KPALPeripheralError(f"Peripheral {peripheral_name} does not exist")

        if attribute_name is None:
            return peripheral, None

        if attribute := peripheral.attributes.get(attribute_name) is None:
            raise KPALAttributeError(
                f"Attribute {attribute_name} does not exist for peripheral {peripheral_name}"
            )

        return peripheral, attribute

    async def create_peripheral(self, peripheral: Peripheral, name: str, *args, **kwargs):
        """Creates a new peripheral instance."""
        if name in self.peripherals:
            raise KPALPeripheralError(
                "Cannot create peripheral with name '{name}' because the name already exists"
            )

        p = await peripheral.build(args, kwargs)

        self.peripherals[name] = p

    async def get_attribute(self, peripheral_name: str, attribute_name: str) -> Value:
        """Gets the value of a peripheral attribute."""
        _, attribute = self.resolve_names(peripheral_name, attribute_name)

        response = await attribute.getter()

        return response

    async def set_attribute(self, value: Value, peripheral_name: str, attribute_name: str) -> None:
        """Sets the value of a peripheral attribute."""
        _, attribute = self.resolve_names(peripheral_name, attribute_name)

        await attribute.setter(value)
