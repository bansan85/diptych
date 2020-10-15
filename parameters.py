from __future__ import annotations
from typing import Union, Tuple
import types


class ErodeParameters:
    class Impl(types.SimpleNamespace):
        size: Tuple[int, int]
        iterations: int

    def __init__(
        self: ErodeParameters, size: Tuple[int, int], iterations: int
    ):
        self.__param = ErodeParameters.Impl(size=size, iterations=iterations)

    @property
    def size(self: ErodeParameters) -> Tuple[int, int]:
        return self.__param.size

    @size.setter
    def size(self: ErodeParameters, val: Tuple[int, int]) -> None:
        self.__param.size = val

    @property
    def iterations(self: ErodeParameters) -> int:
        return self.__param.iterations

    @iterations.setter
    def iterations(self: ErodeParameters, val: int) -> None:
        self.__param.iterations = val

    def init_default_values(
        self: ErodeParameters,
        key: str,
        value: Union[int, float, Tuple[int, int]],
    ) -> None:
        if key == "Size" and isinstance(value, tuple):
            self.size = value
        elif key == "Iterations" and isinstance(value, int):
            self.iterations = value
        else:
            raise Exception("Invalid property.", key)


class CannyParameters:
    class Impl(types.SimpleNamespace):
        minimum: int
        maximum: int
        aperture_size: int

    def __init__(
        self: CannyParameters, minimum: int, maximum: int, aperture_size: int
    ):
        self.__param = CannyParameters.Impl(
            minimum=minimum, maximum=maximum, aperture_size=aperture_size
        )

    @property
    def minimum(self: CannyParameters) -> int:
        return self.__param.minimum

    @minimum.setter
    def minimum(self: CannyParameters, val: int) -> None:
        self.__param.minimum = val

    @property
    def maximum(self: CannyParameters) -> int:
        return self.__param.maximum

    @maximum.setter
    def maximum(self: CannyParameters, val: int) -> None:
        self.__param.maximum = val

    @property
    def aperture_size(self: CannyParameters) -> int:
        return self.__param.aperture_size

    @aperture_size.setter
    def aperture_size(self: CannyParameters, val: int) -> None:
        self.__param.aperture_size = val

    def init_default_values(
        self: CannyParameters,
        key: str,
        value: Union[int, float, Tuple[int, int]],
    ) -> None:
        if key == "Min" and isinstance(value, int):
            self.minimum = value
        elif key == "Max" and isinstance(value, int):
            self.maximum = value
        elif key == "ApertureSize" and isinstance(value, int):
            self.aperture_size = value
        else:
            raise Exception("Invalid property.", key)


class HoughLinesParameters:
    class Impl(types.SimpleNamespace):
        delta_rho: int
        delta_tetha: float
        threshold: int
        min_line_length: int
        max_line_gap: int

    def __init__(  # pylint: disable=too-many-arguments
        self: HoughLinesParameters,
        delta_rho: int,
        delta_tetha: float,
        threshold: int,
        min_line_length: int,
        max_line_gap: int,
    ):
        self.__param = HoughLinesParameters.Impl(
            delta_rho=delta_rho,
            delta_tetha=delta_tetha,
            threshold=threshold,
            min_line_length=min_line_length,
            max_line_gap=max_line_gap,
        )

    @property
    def delta_rho(self: HoughLinesParameters) -> int:
        return self.__param.delta_rho

    @delta_rho.setter
    def delta_rho(self: HoughLinesParameters, val: int) -> None:
        self.__param.delta_rho = val

    @property
    def delta_tetha(self: HoughLinesParameters) -> float:
        return self.__param.delta_tetha

    @delta_tetha.setter
    def delta_tetha(self: HoughLinesParameters, val: float) -> None:
        self.__param.delta_tetha = val

    @property
    def threshold(self: HoughLinesParameters) -> int:
        return self.__param.threshold

    @threshold.setter
    def threshold(self: HoughLinesParameters, val: int) -> None:
        self.__param.threshold = val

    @property
    def min_line_length(self: HoughLinesParameters) -> int:
        return self.__param.min_line_length

    @min_line_length.setter
    def min_line_length(self: HoughLinesParameters, val: int) -> None:
        self.__param.min_line_length = val

    @property
    def max_line_gap(self: HoughLinesParameters) -> int:
        return self.__param.max_line_gap

    @max_line_gap.setter
    def max_line_gap(self: HoughLinesParameters, val: int) -> None:
        self.__param.max_line_gap = val

    def init_default_values(
        self: HoughLinesParameters,
        key: str,
        value: Union[int, float, Tuple[int, int]],
    ) -> None:
        if key == "DeltaRho" and isinstance(value, int):
            self.delta_rho = value
        elif key == "DeltaTetha" and isinstance(value, float):
            self.delta_tetha = value
        elif key == "Threshold" and isinstance(value, int):
            self.threshold = value
        elif key == "MinLineLength" and isinstance(value, int):
            self.min_line_length = value
        elif key == "MaxLineGap" and isinstance(value, int):
            self.max_line_gap = value
        else:
            raise Exception("Invalid property.", key)
