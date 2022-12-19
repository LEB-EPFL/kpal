import pytest


def test_build(serial_peripheral):
    build_args = serial_peripheral.build_args()
    print(build_args)
    raise NotImplementedError
