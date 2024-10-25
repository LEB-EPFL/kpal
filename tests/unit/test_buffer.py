import pytest

from leb.kpal import PixelFormat, RingBuffer


def test_ring_buffer_correct_shape() -> None:
    try:
        RingBuffer(10, (16, 16), PixelFormat.MONO16)
    except ValueError:
        pytest.fail("Failed to instantiate RingBuffer")


def test_ring_buffer_incorrect_shape() -> None:
    with pytest.raises(ValueError):
        RingBuffer(3, (7, 7), PixelFormat.MONO12P)


def test_ring_buffer_size() -> None:
    ring_buffer = RingBuffer(10, (8, 8), PixelFormat.MONO12P)
    expected_size_bytes = (10 * 8 * 8 * 12) // 8

    assert expected_size_bytes == ring_buffer.buf.nbytes
