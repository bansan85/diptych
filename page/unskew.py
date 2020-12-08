import types
from typing import Union, Tuple, List
import numpy as np
import cv2

from parameters import ErodeParameters, CannyParameters, HoughLinesParameters
import cv2ext
from page.find_images import FindImageParameters
import page.find_images
import compute
from debug_image import DebugImage


class UnskewPageParameters:
    class Impl(types.SimpleNamespace):
        erode: ErodeParameters = ErodeParameters((2, 2), 7)
        canny: CannyParameters = CannyParameters(25, 225, 5)
        hough_lines: HoughLinesParameters = HoughLinesParameters(
            1, np.pi / (180 * 20), 70, 300, 90
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
        else:
            raise Exception("Invalid property.", key)


def found_angle_unskew_page(
    lines: List[np.ndarray], delta_angle: float, approximate_angle: float
) -> float:
    histogram = np.zeros(int(np.ceil(90 / delta_angle)) + 1)
    for line in lines:
        angle = (
            np.arctan2(line[3] - line[1], line[2] - line[0]) / np.pi * 180
            + 180
        ) % 90
        i = int(round(angle / delta_angle))
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
                    (x * delta_angle - approximate_angle + 45) % 90 - 45,
                    0,
                    10.0,
                )
                - 0.5
            )
        )
        * histogram_blur[x],
        reverse=True,
    )

    retval = histogram_blur_top_sorted[0] * delta_angle
    if retval >= 45.0:
        retval = retval - 90.0
    return retval


def find_rotation(
    image: np.ndarray,
    parameters: UnskewPageParameters,
    approximate_angle: float,
    debug: DebugImage,
) -> float:
    debug.image(image, DebugImage.Level.DEBUG)

    images_mask = page.find_images.find_images(
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
    eroded2 = eroded & 0b11100000
    debug.image(eroded2, DebugImage.Level.DEBUG)

    # Aide à la détection des contours
    canny = cv2.Canny(
        eroded2,
        parameters.canny.minimum,
        parameters.canny.maximum,
        apertureSize=parameters.canny.aperture_size,
    )
    debug.image(canny, DebugImage.Level.DEBUG)
    canny_filtered = cv2.bitwise_and(canny, cv2.bitwise_not(images_mask))
    debug.image(canny_filtered, DebugImage.Level.DEBUG)

    # Détection des lignes.
    # La précision doit être de l'ordre de 0.05°
    list_lines = cv2.HoughLinesP(
        canny_filtered,
        parameters.hough_lines.delta_rho,
        parameters.hough_lines.delta_tetha,
        parameters.hough_lines.threshold,
        minLineLength=parameters.hough_lines.min_line_length,
        maxLineGap=parameters.hough_lines.max_line_gap,
    )

    # lines contient une liste de liste de lignes.
    # Le deuxième niveau de liste ne contient toujours qu'une ligne.
    lines = list(map(lambda line: line[0], list_lines))

    def image_with_lines() -> np.ndarray:
        retval = cv2ext.convertion_en_couleur(image)
        for line_x1, line_y1, line_x2, line_y2 in lines:
            cv2.line(
                retval,
                (line_x1, line_y1),
                (line_x2, line_y2),
                (255, 0, 0),
                1,
            )
        return retval

    debug.image_lazy(image_with_lines, DebugImage.Level.DEBUG)

    def img() -> np.ndarray:
        retval = cv2ext.convertion_en_couleur(images_mask)
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
        parameters.hough_lines.delta_tetha / np.pi * 180.0,
        approximate_angle,
    )
