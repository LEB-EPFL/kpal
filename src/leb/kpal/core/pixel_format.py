from enum import StrEnum, auto


class PixelFormat(StrEnum):
    MONO12P = auto()
    MONO16 = auto()

    def bits_per_pixel(self) -> int:
        if self == PixelFormat.MONO12P:
            return 12
        if self == PixelFormat.MONO16:
            return 16
        else:
            raise ValueError(f"Unsupported pixel format: {self}")
