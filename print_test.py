import copy
import unittest


class PrintTest:
    def __init__(self, values_to_check):
        self.__values = copy.deepcopy(values_to_check)
        self.__test = unittest.TestCase()

    def print(self, name, data):
        if name in self.__values:
            print("Checking", name, data)
            if self.__values[name][1] == "ecart":
                self.__test.assertGreaterEqual(
                    data, self.__values[name][0] - self.__values[name][2]
                )
                self.__test.assertLessEqual(
                    data, self.__values[name][0] + self.__values[name][2]
                )
            elif self.__values[name][1] == "pourcent":
                self.__test.assertGreaterEqual(
                    data, self.__values[name][0] - data * self.__values[name][2]
                )
                self.__test.assertLessEqual(
                    data, self.__values[name][0] + data * self.__values[name][2]
                )
            else:
                raise Exception("Unknown approximate method", data[1])
        else:
            print("NO CHECK", name, data)
