"""Module that print a tuple (key, value) in release mode.
"""

from typing import Union

from print_interface import PrintInterface


class PrintRelease(PrintInterface):
    """Interface to print a tuple (key, value).

    This is the release implementation. It just shows the key and the value.

    Args:
        PrintInterface (PrintInterface): the print interface
    """

    def print(self, name: str, data: Union[int, float]) -> None:
        """Print the tuple (key, value).

        Args:
            name (str): the name of the key.
            data (Union[int, float]): the data to show.
        """
        print(name, data)

    def close(self) -> None:
        pass
