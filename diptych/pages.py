import types
from typing import Dict, Optional, Tuple, Union

from .angle import Angle
from .crop import CropAroundDataInPageParameters
from .split import SplitTwoWavesParameters
from .unskew import UnskewPageParameters


class Parameters:
    class Impl(types.SimpleNamespace):
        split_two_waves: SplitTwoWavesParameters = SplitTwoWavesParameters()
        unskew_page: UnskewPageParameters = UnskewPageParameters()
        crop_around_data_in_page: CropAroundDataInPageParameters = (
            CropAroundDataInPageParameters()
        )

    def __init__(self) -> None:
        self.__param = Parameters.Impl()

    @property
    def split_two_waves(self) -> SplitTwoWavesParameters:
        return self.__param.split_two_waves

    @property
    def unskew_page(self) -> UnskewPageParameters:
        return self.__param.unskew_page

    @property
    def crop_around_data_in_page(
        self,
    ) -> CropAroundDataInPageParameters:
        return self.__param.crop_around_data_in_page


def init_default_values(
    default_values: Optional[
        Dict[str, Union[int, float, Tuple[int, int], Angle]]
    ]
) -> Parameters:
    parameters = Parameters()

    if default_values is not None:
        for param, value in default_values.items():
            if param.startswith("SplitTwoWaves"):
                parameters.split_two_waves.init_default_values(
                    param[len("SplitTwoWaves") :], value
                )
            elif param.startswith("UnskewPage"):
                parameters.unskew_page.init_default_values(
                    param[len("UnskewPage") :], value
                )
            elif param.startswith("CropAroundDataInPage"):
                parameters.crop_around_data_in_page.init_default_values(
                    param[len("CropAroundDataInPage") :], value
                )
            else:
                raise Exception("Invalid property.", param)

    return parameters
