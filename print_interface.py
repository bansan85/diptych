"""Module that define the interface to print a tuple (key, value).
"""

from abc import ABCMeta, abstractmethod
from typing import Union


class PrintInterface(metaclass=ABCMeta):
    """Interface to print a tuple (key, value).

    The goal of this interface is to be have a way to check if some control
    value is the one expected.

    Args:
        metaclass (ABCMeta, optional): to say this class has abstract methods.
        Defaults to ABCMeta.
    """

    @abstractmethod
    def print(self, name: str, data: Union[int, float]) -> None:
        """Print the tuple (key, value).

        Args:
            name (str): the name of the key.
            data (Union[int, float]): the data to show.
        """

    @abstractmethod
    def close(self) -> None:
        pass


class ConstString:
    @staticmethod
    def separation_double_page_angle() -> str:
        return "separation double page angle"

    @staticmethod
    def separation_double_page_y() -> str:
        return "separation double page y=0"

    @staticmethod
    def page_rotation(n_page: int) -> str:
        return "page rotation " + str(n_page)

    @staticmethod
    def image_crop(n_page: int, zone: str) -> str:
        return "image " + str(n_page) + " crop " + zone

    @staticmethod
    def image_dpi(n_page: int) -> str:
        return "image " + str(n_page) + " dpi"

    @staticmethod
    def image_border(n_page: int, zone: int) -> str:
        if zone == 1:
            return "image " + str(n_page) + " border top"
        if zone == 2:
            return "image " + str(n_page) + " border bottom"
        if zone == 3:
            return "image " + str(n_page) + " border left"
        if zone == 4:
            return "image " + str(n_page) + " border right"
        raise ValueError("zone (" + str(zone) + ") is invalid")
