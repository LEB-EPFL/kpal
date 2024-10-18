from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, auto

import numpy as np

type RawImageShape = tuple[int, int]


@dataclass(frozen=True)
class RawImage:
    """A raw image is a 2D array of (possibly packed) pixel data in a specified format."""

    data: np.ndarray
    pixel_format: PixelFormat

    def __post_init__(self) -> None:
        if self.data.ndim != 2:
            raise ValueError(f"RawImage data must have 2, but got {self.data.ndim}.")

    @property
    def num_cols(self) -> int:
        return self.data.shape[1]

    @property
    def num_rows(self) -> int:
        return self.data.shape[0]

    @property
    def shape(self) -> RawImageShape:
        return self.data.shape


class PixelFormat(StrEnum):
    MONO16 = auto()
