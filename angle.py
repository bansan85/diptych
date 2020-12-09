from __future__ import annotations
from enum import Enum
from typing import Union

import numpy as np


class Angle:
    class Unite(Enum):
        RADIAN = 0
        DEGRE = 1

    def __init__(self, angle: float, unite: Unite):
        if unite == self.Unite.RADIAN:
            self.__angle = angle
        else:
            self.__angle = angle / 180.0 * np.pi

    def get_rad(self) -> float:
        return self.__angle

    def get_deg(self) -> float:
        return self.__angle / np.pi * 180.0

    # Binary Operators
    def __add__(self, other: Angle) -> Angle:
        return Angle(self.get_rad() + other.get_rad(), Angle.Unite.RADIAN)

    def __sub__(self, other: Angle) -> Angle:
        return Angle(self.get_rad() - other.get_rad(), Angle.Unite.RADIAN)

    def __mul__(self, other: Union[float, int]) -> Angle:
        return Angle(self.get_rad() * other, Angle.Unite.RADIAN)

    def __truediv__(self, other: Angle) -> float:
        return self.get_rad() / other.get_rad()

    def __mod__(self, other: Angle) -> Angle:
        return Angle(self.get_rad() % other.get_rad(), Angle.Unite.RADIAN)

    # Comparison Operators
    def __lt__(self, other: Angle) -> bool:
        return self.get_rad() < other.get_rad()

    def __gt__(self, other: Angle) -> bool:
        return self.get_rad() > other.get_rad()

    def __le__(self, other: Angle) -> bool:
        return self.get_rad() <= other.get_rad()

    def __ge__(self, other: Angle) -> bool:
        return self.get_rad() >= other.get_rad()

    # Unary Operators
    def __neg__(self) -> Angle:
        return Angle(-self.get_rad(), Angle.Unite.RADIAN)

    def __repr__(self) -> str:
        return str(self.get_deg()) + "°"

    def __str__(self) -> str:
        return str(self.get_deg())

    @staticmethod
    def abs(angle: Angle) -> Angle:
        return Angle(np.abs(angle.get_rad()), Angle.Unite.RADIAN)

    @staticmethod
    def rad(angle: float) -> Angle:
        return Angle(angle, Angle.Unite.RADIAN)

    @staticmethod
    def deg(angle: float) -> Angle:
        return Angle(angle, Angle.Unite.DEGRE)

    # angle is stored in radian
    __angle: float
