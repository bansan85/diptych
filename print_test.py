"""Module that print a tuple (key, value) in test mode.
"""

from __future__ import annotations
import copy
import time
from typing import Any, Dict, Union, Optional
import unittest

from print_interface import PrintInterface


class PrintTest(PrintInterface):
    """Interface to print a tuple (key, value).

    This is the test implementation. It shows the key and the value and
    check if the value is in a valid bondary.

    Args:
        PrintInterface (PrintInterface): the print interface
    """

    __last_msg_assert: Optional[str]

    def __init__(self, values_to_check: Dict[str, Any]):
        """Constructor for test purpose.

        You tell to the print instance the value expected for each key.

        The value must be a list :
            - first : the expected value,
            - second : the type of comparison :
                - difference : the value must be between
                    v-difference and v+difference.
                - pourcentage : in float représentation. The value must be
                    between v*(1-pourcentage) and v*(1+pourcentage).
            - third : the value of difference of pourcentage used by the
                previous argumment.

            values_to_check (Dict[str, Any]): a list of key, value association.
        """
        self.__values = copy.deepcopy(values_to_check)
        self.__test = unittest.TestCase()
        self.__last_msg_assert = None

    def __check(
        self,
        name: str,
        value: Union[int, float],
        minimum: Union[int, float],
        maximum: Union[int, float],
    ) -> None:
        try:
            self.__test.assertGreaterEqual(value, minimum)
            self.__test.assertLessEqual(value, maximum)
        except AssertionError as err:
            self.__last_msg_assert = "{0} : {1}".format(name, err)
            print(self.__last_msg_assert)

    def print(self, name: str, data: Union[int, float]) -> None:
        """Print the tuple (key, value) and check if the value is in the
        expected boundary.

        If the key doesn't exist in the dictionary when calling print,
        the value is just print with a warning "NO CHECK".

        Args:
            name (str): the name of the key
            data (Union[int, float]): the value to check.

        Raises:
            ValueError: raised if the value is outside of the valid domain.
        """
        if name in self.__values:
            print(str(time.time_ns()) + " : Checking", name, data)
            if self.__values[name][1] == "difference":
                self.__check(
                    name,
                    data,
                    self.__values[name][0] - self.__values[name][2],
                    self.__values[name][0] + self.__values[name][2],
                )
            elif self.__values[name][1] == "range":
                self.__check(
                    name,
                    data,
                    self.__values[name][0] + self.__values[name][2],
                    self.__values[name][0] + self.__values[name][3],
                )
            elif self.__values[name][1] == "pourcent":
                self.__check(
                    name,
                    data,
                    self.__values[name][0] - data * self.__values[name][2],
                    self.__values[name][0] + data * self.__values[name][2],
                )
            else:
                raise ValueError(
                    "Unknown approximate method", self.__values[name][1]
                )
        else:
            print(str(time.time_ns()) + " : NO CHECK", name, data)

    def close(self) -> None:
        if self.__last_msg_assert is not None:
            raise Exception(self.__last_msg_assert)
