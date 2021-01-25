import types
from typing import List, Tuple, Union

import cv2
import numpy as np

from . import compute, cv2ext, find_images
from .angle import Angle
from .debug_image import DebugImage
from .find_images import FindImageParameters
from .parameters import CannyParameters, ErodeParameters, HoughLinesParameters


class UnskewPageParameters:
    class Impl(types.SimpleNamespace):
        erode: ErodeParameters = ErodeParameters((2, 2), 7)
        canny: CannyParameters = CannyParameters(25, 225, 5)
        hough_lines: HoughLinesParameters = HoughLinesParameters(
            1, Angle.deg(1 / 20), 100, 300, 30, 0.35
        )
        find_images: FindImageParameters = FindImageParameters(
            5,
            (10, 10),
            (10, 10),
            (10, 10),
            0.01,
        )

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
    def find_images(self) -> FindImageParameters:
        return self.__param.find_images

    def init_default_values(
        self,
        key: str,
        value: Union[int, float, Tuple[int, int], Angle],
    ) -> None:
        if key.startswith("Erode"):
            self.erode.init_default_values(key[len("Erode") :], value)
        elif key.startswith("Canny"):
            self.canny.init_default_values(key[len("Canny") :], value)
        elif key.startswith("HoughLines"):
            self.hough_lines.init_default_values(
                key[len("HoughLines") :], value
            )
        else:
            raise Exception("Invalid property.", key)


def found_angle_unskew_page(
    lines: List[np.ndarray], delta_angle: Angle, approximate_angle: Angle
) -> Angle:
    histogram = np.zeros(int(np.ceil(90 / delta_angle.get_deg())) + 1)
    for line in lines:
        angle = (
            np.arctan2(line[3] - line[1], line[2] - line[0]) / np.pi * 180
            + 180
        ) % 90
        i = int(round(angle / delta_angle.get_deg()))
        length = np.linalg.norm(
            np.array((line[0], line[1])) - np.array((line[2], line[3]))
        )
        histogram[i] = histogram[i] + length

    histogram_blur = cv2ext.gaussian_blur_wrap(histogram, 9)
    histogram_blur_top = compute.get_tops_indices_histogram(histogram_blur)
    histogram_blur_top_sorted = sorted(
        histogram_blur_top,
        key=lambda x: (
            1.0
            - 2.0
            * np.absolute(
                compute.norm_cdf(
                    (
                        delta_angle * x - approximate_angle + Angle.deg(45)
                    ).get_deg()
                    % 90.0
                    - 45.0,
                    0,
                    10.0,
                )
                - 0.5
            )
        )
        * histogram_blur[x],
        reverse=True,
    )

    retval = delta_angle * histogram_blur_top_sorted[0]
    if retval >= Angle.deg(45):
        retval = retval - Angle.deg(90)
    return retval


def find_rotation(
    image: np.ndarray,
    parameters: UnskewPageParameters,
    approximate_angle: Angle,
    debug: DebugImage,
) -> Angle:
    debug.image(image, DebugImage.Level.DEBUG)

    images_mask = find_images.find_images(
        image,
        parameters.find_images,
        approximate_angle,
        debug,
    )

    img_gauche = cv2ext.convertion_en_niveau_de_gris(image)
    # On grossit les images pour former un boudin et mieux détecter les
    # lignes.
    eroded = cv2.erode(
        img_gauche,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, parameters.erode.size),
        iterations=parameters.erode.iterations,
    )
    debug.image(eroded, DebugImage.Level.DEBUG)

    small = cv2.resize(
        eroded,
        (0, 0),
        fx=parameters.hough_lines.scale,
        fy=parameters.hough_lines.scale,
    )

    eroded2 = small & 0b11100000
    debug.image(eroded2, DebugImage.Level.DEBUG)

    # Aide à la détection des contours
    canny = cv2.Canny(
        eroded2,
        parameters.canny.minimum,
        parameters.canny.maximum,
        apertureSize=parameters.canny.aperture_size,
    )
    debug.image(canny, DebugImage.Level.DEBUG)
    canny_filtered = cv2.bitwise_and(
        canny,
        cv2.resize(
            cv2.bitwise_not(images_mask),
            (0, 0),
            fx=parameters.hough_lines.scale,
            fy=parameters.hough_lines.scale,
        ),
    )
    debug.image(canny_filtered, DebugImage.Level.DEBUG)

    # Détection des lignes.
    # La précision doit être de l'ordre de 0.05°
    list_lines = cv2.HoughLinesP(
        canny_filtered,
        parameters.hough_lines.delta_rho,
        parameters.hough_lines.delta_tetha.get_rad(),
        parameters.hough_lines.threshold,
        minLineLength=parameters.hough_lines.min_line_length,
        maxLineGap=parameters.hough_lines.max_line_gap,
    )

    if list_lines is None:
        return Angle.rad(0.0)

    # lines contient une liste de liste de lignes.
    # Le deuxième niveau de liste ne contient toujours qu'une ligne.
    lines = list(map(lambda line: line[0], list_lines))

    def image_with_lines() -> np.ndarray:
        retval = cv2.resize(
            cv2ext.convertion_en_couleur(image),
            (0, 0),
            fx=parameters.hough_lines.scale,
            fy=parameters.hough_lines.scale,
        )
        for line_x1, line_y1, line_x2, line_y2 in lines:
            cv2.line(
                retval,
                (line_x1, line_y1),
                (line_x2, line_y2),
                (255, 0, 0),
                1,
            )
        return retval

    debug.image_lazy(image_with_lines, DebugImage.Level.INFO)

    def img() -> np.ndarray:
        retval = cv2.resize(
            cv2ext.convertion_en_couleur(images_mask),
            (0, 0),
            fx=parameters.hough_lines.scale,
            fy=parameters.hough_lines.scale,
        )
        for line_x1, line_y1, line_x2, line_y2 in lines:
            cv2.line(
                retval,
                (line_x1, line_y1),
                (line_x2, line_y2),
                (255, 0, 0),
                1,
            )
        return retval

    debug.image_lazy(img, DebugImage.Level.DEBUG)

    return found_angle_unskew_page(
        lines,
        parameters.hough_lines.delta_tetha,
        approximate_angle,
    )
