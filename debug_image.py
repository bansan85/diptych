from __future__ import annotations
from enum import IntEnum
from typing import List, Any
from collections.abc import Callable


import numpy as np


import cv2ext


class DebugImage:
    class Level(IntEnum):
        OFF = 4

        # Most important image to check if algorithm works well
        TOP = 3

        # If top fails, interesting images to understand where
        # algorithm fails.
        INFO = 2

        # All images
        DEBUG = 1

    def __init__(self, level: Level = Level.OFF, root: str = "") -> None:
        self.__level = level
        self.__root = root
        self.__status = [1]
        self.__sub_level = 0
        self.__next_when_dec = [False]

    def set_root(self, root: str) -> None:
        self.__root = root

    def inc(self) -> DebugImage:
        self.__sub_level = self.__sub_level + 1
        if len(self.__status) < self.__sub_level + 1:
            self.__status.append(1)
        if len(self.__next_when_dec) < self.__sub_level + 1:
            self.__next_when_dec.append(False)
        return self

    def dec(self) -> None:
        self.__sub_level = self.__sub_level - 1
        if self.__next_when_dec[self.__sub_level]:
            self.__next_when_dec[self.__sub_level] = False
            self.__next()

    def __next(self) -> None:
        self.__status[self.__sub_level] += 1
        self.__status[self.__sub_level + 1 :] = [
            1 for x in self.__status[self.__sub_level + 1 :]
        ]

    def name(self) -> str:
        retval = self.__root
        for i in range(self.__sub_level + 1):
            retval = retval + "_" + str(self.__status[i])
        return retval + ".png"

    def image(self, img: np.ndarray, level: Level) -> None:
        if level >= self.__level:
            name = self.name()
            self.__next()
            self.__next_when_dec[0 : self.__sub_level] = [
                True for x in range(self.__sub_level)
            ]
            cv2ext.secure_write(name, img)

    def image_lazy(self, func: Callable[[], np.ndarray], level: Level) -> None:
        if level >= self.__level:
            name = self.name()
            self.__next()
            self.__next_when_dec[0 : self.__sub_level] = [
                True for x in range(self.__sub_level)
            ]
            cv2ext.secure_write(name, func())

    __level: Level
    __root: str
    __status: List[int]
    __sub_level: int
    __next_when_dec: List[bool]


def inc_debug(function_to_decorate: Any) -> Any:
    def wrapper(*args: Any) -> Any:
        for arg in args:
            if isinstance(arg, DebugImage):
                arg.inc()
                try:
                    return function_to_decorate(*args)
                finally:
                    arg.dec()
        raise Exception("Failed to found DebugImage in inc_debug decorator")

    return wrapper
