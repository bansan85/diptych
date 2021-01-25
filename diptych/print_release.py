"""Module that print a tuple (key, value) in release mode.
"""

from .print_interface import _N, PrintInterface


class PrintRelease(PrintInterface):
    """Interface to print a tuple (key, value).

    This is the release implementation. It just shows the key and the value.

    Args:
        PrintInterface (PrintInterface): the print interface
    """

    def print(self, name: str, data: _N) -> None:
        """Print the tuple (key, value).

        Args:
            name (str): the name of the key.
            data (Union[int, float]): the data to show.
        """
        print(name, data)

    def close(self) -> None:
        pass
