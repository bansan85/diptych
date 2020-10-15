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
