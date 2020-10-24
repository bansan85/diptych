import types
from typing import Union, Tuple, Any, Optional
import numpy as np
import cv2

from parameters import ErodeParameters, CannyParameters, HoughLinesParameters
import cv2ext
from page.find_images import FindImageParameters
import page.find_images
import compute


class UnskewPageParameters:
    class Impl(types.SimpleNamespace):
        erode: ErodeParameters = ErodeParameters((2, 2), 7)
        canny: CannyParameters = CannyParameters(25, 225, 5)
        hough_lines: HoughLinesParameters = HoughLinesParameters(
            1, np.pi / (180 * 20), 70, 300, 90
        )
        find_images: FindImageParameters = FindImageParameters(
            0.005,
            (10, 10),
            (10, 10),
            (10, 10),
            8,
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


def found_angle_unskew_page(lines: Any, delta_angle: float) -> Any:
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
    histogram_blur = cv2.GaussianBlur(
        histogram, (9, 9), 9, 9, cv2.BORDER_REPLICATE
    )

    return np.argmax(histogram_blur) * delta_angle


def find_rotation(
    image: Any,
    n_page: int,
    parameters: UnskewPageParameters,
    enable_debug: Optional[str],
) -> float:
    cv2ext.write_image_if(image, enable_debug, "_" + str(n_page) + "_1.png")

    images_mask = page.find_images.find_images(
        image,
        parameters.find_images,
        compute.optional_concat(
            enable_debug, "_" + str(n_page) + "_1______.png"
        ),
    )

    img_gauche = cv2ext.convertion_en_niveau_de_gris(image)
    # On grossit les images pour former un boudin et mieux détecter les
    # lignes.
    eroded = cv2.erode(
        img_gauche,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, parameters.erode.size),
        iterations=parameters.erode.iterations,
    )
    cv2ext.write_image_if(eroded, enable_debug, "_" + str(n_page) + "_2.png")

    # Aide à la détection des contours
    canny = cv2.Canny(
        eroded,
        parameters.canny.minimum,
        parameters.canny.maximum,
        apertureSize=parameters.canny.aperture_size,
    )
    cv2ext.write_image_if(canny, enable_debug, "_" + str(n_page) + "_3.png")

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

    if enable_debug is not None:
        image_with_lines = cv2ext.convertion_en_couleur(image)
        img = cv2ext.convertion_en_couleur(images_mask)
        for line_x1, line_y1, line_x2, line_y2 in lines:
            cv2.line(
                image_with_lines,
                (line_x1, line_y1),
                (line_x2, line_y2),
                (255, 0, 0),
                1,
            )
            cv2.line(
                img,
                (line_x1, line_y1),
                (line_x2, line_y2),
                (255, 0, 0),
                1,
            )
        cv2.imwrite(
            enable_debug + "_" + str(n_page) + "_4.png", image_with_lines
        )
        cv2.imwrite(enable_debug + "_" + str(n_page) + "_4bbbb.png", img)

    lines_filtered = []
    for line_x1, line_y1, line_x2, line_y2 in lines:
        image_line = np.zeros(images_mask.shape, np.uint8)
        cv2.line(
            image_line,
            (line_x1, line_y1),
            (line_x2, line_y2),
            (255, 255, 255),
            1,
        )
        image_line = cv2.bitwise_and(images_mask, image_line)
        if cv2.countNonZero(image_line) == 0:
            lines_filtered.append((line_x1, line_y1, line_x2, line_y2))

    if enable_debug is not None:
        image_with_lines = cv2ext.convertion_en_couleur(image)
        for line_x1, line_y1, line_x2, line_y2 in lines_filtered:
            cv2.line(
                image_with_lines,
                (line_x1, line_y1),
                (line_x2, line_y2),
                (255, 0, 0),
                1,
            )
        cv2.imwrite(
            enable_debug + "_" + str(n_page) + "_4ccccc.png", image_with_lines
        )

    return found_angle_unskew_page(
        lines_filtered,
        parameters.hough_lines.delta_tetha / np.pi * 180.0,
    )
