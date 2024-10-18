import numpy as np

from leb.kpal import PixelFormat, RawImage, RingBuffer


def test_ring_buffer():
    capacity = 3
    shape = (4, 4)
    dtype = np.uint16

    buf = RingBuffer(capacity=capacity, shape=shape, dtype=dtype)

    image1 = RawImage(data=np.ones(shape=shape, dtype=dtype), pixel_format=PixelFormat.MONO16)
    image2 = RawImage(data=2 * np.ones(shape=shape, dtype=dtype), pixel_format=PixelFormat.MONO16)
    image3 = RawImage(data=3 * np.ones(shape=shape, dtype=dtype), pixel_format=PixelFormat.MONO16)
    image4 = RawImage(data=4 * np.ones(shape=shape, dtype=dtype), pixel_format=PixelFormat.MONO16)

    buf.put(image1)
    assert np.all(buf[0] == 1)

    buf.put(image2)
    assert np.all(buf[0] == 1)
    assert np.all(buf[1] == 2)

    buf.put(image3)
    assert np.all(buf[0] == 1)
    assert np.all(buf[1] == 2)
    assert np.all(buf[2] == 3)

    buf.put(image4)  # Overwrite the oldest image
    assert np.all(buf[0] == 4)
    assert np.all(buf[1] == 2)
    assert np.all(buf[2] == 3)
