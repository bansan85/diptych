import types
from typing import Union, Tuple, Any, Optional
import numpy as np
import cv2

from parameters import ErodeParameters, CannyParameters, HoughLinesParameters
import compute
import cv2ext


class UnskewPageParameters:
    class Impl(types.SimpleNamespace):
        erode: ErodeParameters = ErodeParameters((2, 2), 7)
        canny: CannyParameters = CannyParameters(25, 225, 5)
        hough_lines: HoughLinesParameters = HoughLinesParameters(
            1, np.pi / (180 * 20), 70, 300, 90
        )
        angle_limit: float = 20.0
        angle_limit_stddev: float = 0.75

    def __init__(self) -> None:
        self.__param = UnskewPageParameters.Impl()

    @property
    def erode(self) -> ErodeParameters:
        return self.__param.erode

    @property
    def canny(self) -> CannyParameters:
        return self.__param.canny

    @property
    def hough_lines(self) -> HoughLinesParameters:
        return self.__param.hough_lines

    @property
    def angle_limit(self) -> float:
        return self.__param.angle_limit

    @angle_limit.setter
    def angle_limit(self, val: float) -> None:
        self.__param.angle_limit = val

    @property
    def angle_limit_stddev(self) -> float:
        return self.__param.angle_limit_stddev

    @angle_limit_stddev.setter
    def angle_limit_stddev(self, val: float) -> None:
        self.__param.angle_limit_stddev = val

    def init_default_values(
        self,
        key: str,
        value: Union[int, float, Tuple[int, int]],
    ) -> None:
        if key.startswith("Erode"):
            self.erode.init_default_values(key[len("Erode") :], value)
        elif key.startswith("Canny"):
            self.canny.init_default_values(key[len("Canny") :], value)
        elif key.startswith("HoughLines"):
            self.hough_lines.init_default_values(
                key[len("HoughLines") :], value
            )
        elif key == "AngleLimit" and isinstance(value, float):
            self.angle_limit = value
        elif key == "AngleLimitStddev" and isinstance(value, float):
            self.angle_limit_stddev = value
        else:
            raise Exception("Invalid property.", key)


def found_candidates_angle_unskew_page(lines: Any, angle: float) -> Any:
    # TODO: Use histogram to find the angle instead of
    # supposing the angle is closed to 0 +/- limit_angle.
    def is_angle_vertical_or_horizontal(
        line: Tuple[int, int, int, int],
        limit_angle: float = angle,
    ) -> bool:
        point1_x, point1_y, point2_x, point2_y = line
        angl = (
            np.arctan2(point2_y - point1_y, point2_x - point1_x) / np.pi * 180
            + 180
        ) % 90
        return angl < limit_angle or angl > 90 - limit_angle

    return list(filter(is_angle_vertical_or_horizontal, lines))


def found_best_angle_unskew_page(
    valid_lines: Any, angle_limit: float, angle_limit_stddev: float
) -> float:
    # On converti les lignes en angles
    def convert_line_to_angle_closed_to_zero(
        line: Tuple[int, int, int, int], limit_angle: float
    ) -> float:
        angle = compute.get_angle_0_180((line[2], line[3]), (line[0], line[1]))
        if angle > 180 - limit_angle:
            return angle - 180
        if 90 - limit_angle < angle < 90 + limit_angle:
            return angle - 90
        return angle

    angles = list(
        map(
            lambda x: convert_line_to_angle_closed_to_zero(x, angle_limit),
            valid_lines,
        )
    )

    # On enlève les valeurs extrêmes
    ecarttype = np.std(angles) * angle_limit_stddev
    moyenne = np.mean(angles)
    angle_dans_ecarttype = list(
        filter(
            lambda x: moyenne - ecarttype < x < moyenne + ecarttype,
            angles,
        )
    )

    if len(angle_dans_ecarttype) == 0:
        raise Exception(
            "No angle have the defined constraints.",
            "AngleLimitStddev too small ?",
        )
    return np.mean(angle_dans_ecarttype)


def find_rotation(
    image: Any,
    n_page: int,
    parameters: UnskewPageParameters,
    enable_debug: Optional[str],
) -> float:
    if enable_debug is not None:
        cv2.imwrite(enable_debug + "_" + str(n_page) + "_1.png", image)
    img_gauche = cv2ext.convertion_en_niveau_de_gris(image)
    # On grossit les images pour former un boudin et mieux détecter les
    # lignes.
    eroded = cv2.erode(
        img_gauche,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, parameters.erode.size),
        iterations=parameters.erode.iterations,
    )
    if enable_debug is not None:
        cv2.imwrite(enable_debug + "_" + str(n_page) + "_2.png", eroded)

    # Aide à la détection des contours
    canny = cv2.Canny(
        eroded,
        parameters.canny.minimum,
        parameters.canny.maximum,
        apertureSize=parameters.canny.aperture_size,
    )
    if enable_debug is not None:
        cv2.imwrite(enable_debug + "_" + str(n_page) + "_3.png", canny)

    # Détection des lignes.
    # La précision doit être de l'ordre de 0.05°
    list_lines = cv2.HoughLinesP(
        canny,
        parameters.hough_lines.delta_rho,
        parameters.hough_lines.delta_tetha,
        parameters.hough_lines.threshold,
        minLineLength=parameters.hough_lines.min_line_length,
        maxLineGap=parameters.hough_lines.max_line_gap,
    )

    # lines contient une liste de liste de lignes.
    # Le deuxième niveau de liste ne contient toujours qu'une ligne.
    lines = list(map(lambda line: line[0], list_lines))
    valid_lines = found_candidates_angle_unskew_page(
        lines, parameters.angle_limit
    )

    if enable_debug is not None:
        image_with_lines = image.copy()
        for line_x1, line_y1, line_x2, line_y2 in lines:
            cv2.line(
                image_with_lines,
                (line_x1, line_y1),
                (line_x2, line_y2),
                (255, 0, 0),
                1,
            )
        for line_x1, line_y1, line_x2, line_y2 in valid_lines:
            cv2.line(
                image_with_lines,
                (line_x1, line_y1),
                (line_x2, line_y2),
                (0, 0, 255),
                1,
            )
        cv2.imwrite(
            enable_debug + "_" + str(n_page) + "_4.png", image_with_lines
        )

    return found_best_angle_unskew_page(
        valid_lines, parameters.angle_limit, parameters.angle_limit_stddev
    )
