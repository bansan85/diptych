from __future__ import annotations

from copy import deepcopy
from enum import Enum
import types
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from angle import Angle
import compute
import cv2ext
from debug_image import DebugImage, inc_debug
import ocr
import page.find_images
from page.find_images import FindImageParameters
from parameters import CannyParameters, ErodeParameters, HoughLinesParameters


class FoundDataTry1Parameters:
    class Impl(types.SimpleNamespace):
        erode: ErodeParameters = ErodeParameters((9, 9), 1)
        threshold: int = 240
        pourcentage_ecart_rectangle: float = 10.0
        hough_lines: HoughLinesParameters = HoughLinesParameters(
            1, Angle.deg(1 / 20), 30, 100, 30, 1.0
        )

    def __init__(self) -> None:
        self.__param = FoundDataTry1Parameters.Impl()

    @property
    def erode(self) -> ErodeParameters:
        return self.__param.erode

    @property
    def threshold(self) -> int:
        return self.__param.threshold

    @threshold.setter
    def threshold(self, val: int) -> None:
        self.__param.threshold = val

    @property
    def pourcentage_ecart_rectangle(
        self,
    ) -> float:
        return self.__param.pourcentage_ecart_rectangle

    @pourcentage_ecart_rectangle.setter
    def pourcentage_ecart_rectangle(self, val: float) -> None:
        self.__param.pourcentage_ecart_rectangle = val

    @property
    def hough_lines(
        self,
    ) -> HoughLinesParameters:
        return self.__param.hough_lines


class FoundDataTry2Parameters:
    class Impl(types.SimpleNamespace):
        blur_size: Tuple[int, int] = (10, 10)
        threshold_gray: int = 200
        kernel_morpho_size: Tuple[int, int] = (10, 10)
        canny_gray: CannyParameters = CannyParameters(25, 255, 5)
        hough_lines_gray: HoughLinesParameters = HoughLinesParameters(
            1, Angle.deg(1 / 20), 30, 100, 30, 1.0
        )
        threshold_histogram: int = 15
        canny_histogram: CannyParameters = CannyParameters(25, 255, 5)
        hough_lines_histogram: HoughLinesParameters = HoughLinesParameters(
            1, Angle.deg(1 / 20), 30, 100, 30, 1.0
        )
        find_images: FindImageParameters = FindImageParameters(
            5,
            (10, 10),
            (10, 10),
            (10, 10),
            0.01,
        )

    def __init__(self) -> None:
        self.__param = FoundDataTry2Parameters.Impl()

    @property
    def blur_size(self) -> Tuple[int, int]:
        return self.__param.blur_size

    @blur_size.setter
    def blur_size(self, val: Tuple[int, int]) -> None:
        self.__param.blur_size = val

    @property
    def threshold_gray(self) -> int:
        return self.__param.threshold_gray

    @threshold_gray.setter
    def threshold_gray(self, val: int) -> None:
        self.__param.threshold_gray = val

    @property
    def kernel_morpho_size(self) -> Tuple[int, int]:
        return self.__param.kernel_morpho_size

    @kernel_morpho_size.setter
    def kernel_morpho_size(self, val: Tuple[int, int]) -> None:
        self.__param.kernel_morpho_size = val

    @property
    def canny_gray(self) -> CannyParameters:
        return self.__param.canny_gray

    @property
    def hough_lines_gray(
        self,
    ) -> HoughLinesParameters:
        return self.__param.hough_lines_gray

    @property
    def threshold_histogram(self) -> int:
        return self.__param.threshold_histogram

    @threshold_histogram.setter
    def threshold_histogram(self, val: int) -> None:
        self.__param.threshold_histogram = val

    @property
    def canny_histogram(self) -> CannyParameters:
        return self.__param.canny_histogram

    @property
    def hough_lines_histogram(
        self,
    ) -> HoughLinesParameters:
        return self.__param.hough_lines_histogram

    @property
    def find_images(self) -> FindImageParameters:
        return self.__param.find_images


class CropAroundDataInPageParameters:
    class PositionInside(Enum):
        UNKNOWN = 0
        LEFT = 1
        RIGHT = 2

    # pylint: disable=too-many-instance-attributes
    class Impl(types.SimpleNamespace):
        found_data_try1: FoundDataTry1Parameters = FoundDataTry1Parameters()
        found_data_try2: FoundDataTry2Parameters = FoundDataTry2Parameters()
        dilate_size: Tuple[int, int] = (5, 5)
        threshold2: int = 200
        contour_area_min: float = 0.005 * 0.005
        contour_area_max: float = 1.0
        border: int = 10
        closed_to_edge_x_inside_edge_min: float = 0.025
        closed_to_edge_x_inside_edge_max: float = 0.05
        closed_to_edge_x_outside_edge_min: float = 0.05
        closed_to_edge_x_outside_edge_max: float = 0.10
        closed_to_edge_y_min: float = 0.02
        closed_to_edge_y_max: float = 0.05

    def __init__(self) -> None:
        self.__param = CropAroundDataInPageParameters.Impl()

    @property
    def found_data_try1(
        self,
    ) -> FoundDataTry1Parameters:
        return self.__param.found_data_try1

    @property
    def found_data_try2(
        self,
    ) -> FoundDataTry2Parameters:
        return self.__param.found_data_try2

    @property
    def dilate_size(self) -> Tuple[int, int]:
        return self.__param.dilate_size

    @dilate_size.setter
    def dilate_size(self, val: Tuple[int, int]) -> None:
        self.__param.dilate_size = val

    @property
    def threshold2(self) -> int:
        return self.__param.threshold2

    @threshold2.setter
    def threshold2(self, val: int) -> None:
        self.__param.threshold2 = val

    @property
    def contour_area_min(self) -> float:
        return self.__param.contour_area_min

    @contour_area_min.setter
    def contour_area_min(self, val: float) -> None:
        self.__param.contour_area_min = val

    @property
    def contour_area_max(self) -> float:
        return self.__param.contour_area_max

    @contour_area_max.setter
    def contour_area_max(self, val: float) -> None:
        self.__param.contour_area_max = val

    @property
    def border(self) -> int:
        return self.__param.border

    @border.setter
    def border(self, val: int) -> None:
        self.__param.border = val

    @property
    def closed_to_edge_x_inside_edge_min(self) -> float:
        return self.__param.closed_to_edge_x_inside_edge_min

    @closed_to_edge_x_inside_edge_min.setter
    def closed_to_edge_x_inside_edge_min(self, val: float) -> None:
        self.__param.closed_to_edge_x_inside_edge_min = val

    @property
    def closed_to_edge_x_inside_edge_max(self) -> float:
        return self.__param.closed_to_edge_x_inside_edge_max

    @closed_to_edge_x_inside_edge_max.setter
    def closed_to_edge_x_inside_edge_max(self, val: float) -> None:
        self.__param.closed_to_edge_x_inside_edge_max = val

    @property
    def closed_to_edge_x_outside_edge_min(self) -> float:
        return self.__param.closed_to_edge_x_outside_edge_min

    @closed_to_edge_x_outside_edge_min.setter
    def closed_to_edge_x_outside_edge_min(self, val: float) -> None:
        self.__param.closed_to_edge_x_outside_edge_min = val

    @property
    def closed_to_edge_x_outside_edge_max(self) -> float:
        return self.__param.closed_to_edge_x_outside_edge_max

    @closed_to_edge_x_outside_edge_max.setter
    def closed_to_edge_x_outside_edge_max(self, val: float) -> None:
        self.__param.closed_to_edge_x_outside_edge_max = val

    @property
    def closed_to_edge_y_min(self) -> float:
        return self.__param.closed_to_edge_y_min

    @closed_to_edge_y_min.setter
    def closed_to_edge_y_min(self, val: float) -> None:
        self.__param.closed_to_edge_y_min = val

    @property
    def closed_to_edge_y_max(self) -> float:
        return self.__param.closed_to_edge_y_max

    @closed_to_edge_y_max.setter
    def closed_to_edge_y_max(self, val: float) -> None:
        self.__param.closed_to_edge_y_max = val

    def set_pos_inside_left(self) -> CropAroundDataInPageParameters:
        retval = deepcopy(self)
        retval.pos = self.PositionInside.LEFT
        return retval

    def set_pos_inside_right(self) -> CropAroundDataInPageParameters:
        retval = deepcopy(self)
        retval.pos = self.PositionInside.RIGHT
        return retval

    def init_default_values(
        self,
        key: str,
        value: Union[int, float, Tuple[int, int], Angle],
    ) -> None:
        if key.startswith("Erode"):
            self.found_data_try1.erode.init_default_values(
                key[len("Erode") :], value
            )
        elif key == "Threshold1" and isinstance(value, int):
            self.found_data_try1.threshold = value
        elif key == "PourcentageEcartRectangle" and isinstance(value, float):
            self.found_data_try1.pourcentage_ecart_rectangle = value
        elif key == "DilateSize" and isinstance(value, tuple):
            self.dilate_size = value
        elif key == "Threshold2" and isinstance(value, int):
            self.threshold2 = value
        elif key == "ContourAreaMin" and isinstance(value, float):
            self.contour_area_min = value
        elif key == "ContourAreaMax" and isinstance(value, float):
            self.contour_area_max = value
        elif key == "Border" and isinstance(value, int):
            self.border = value
        else:
            raise Exception("Invalid property.", key)

    def get_big_rectangle(
        self, imgw: int, imgh: int
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        if self.pos == self.PositionInside.LEFT:
            return (
                (
                    int(self.closed_to_edge_x_inside_edge_max * imgw),
                    int(self.closed_to_edge_y_max * imgh),
                ),
                (
                    int((1 - self.closed_to_edge_x_outside_edge_max) * imgw),
                    int((1 - self.closed_to_edge_y_max) * imgh),
                ),
            )

        if self.pos == self.PositionInside.RIGHT:
            return (
                (
                    int(self.closed_to_edge_x_outside_edge_max * imgw),
                    int(self.closed_to_edge_y_max * imgh),
                ),
                (
                    int((1 - self.closed_to_edge_x_inside_edge_max) * imgw),
                    int((1 - self.closed_to_edge_y_max) * imgh),
                ),
            )

        raise Exception("PositionInside is unknown")

    def get_small_rectangle(
        self, imgw: int, imgh: int
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        if self.pos == self.PositionInside.LEFT:
            return (
                (
                    int(self.closed_to_edge_x_inside_edge_min * imgw),
                    int(self.closed_to_edge_y_min * imgh),
                ),
                (
                    int((1 - self.closed_to_edge_x_outside_edge_min) * imgw),
                    int((1 - self.closed_to_edge_y_min) * imgh),
                ),
            )

        if self.pos == self.PositionInside.RIGHT:
            return (
                (
                    int(self.closed_to_edge_x_outside_edge_min * imgw),
                    int(self.closed_to_edge_y_min * imgh),
                ),
                (
                    int((1 - self.closed_to_edge_x_inside_edge_min) * imgw),
                    int((1 - self.closed_to_edge_y_min) * imgh),
                ),
            )

        raise Exception("PositionInside is unknown")

    pos: PositionInside = PositionInside.UNKNOWN


@inc_debug
def found_data_try1(
    image: np.ndarray,
    param: FoundDataTry1Parameters,
    debug: DebugImage,
) -> Optional[np.ndarray]:
    gray = cv2ext.convertion_en_niveau_de_gris(image)
    eroded = cv2.erode(
        gray,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, param.erode.size),
        iterations=param.erode.iterations,
    )
    debug.image(eroded, DebugImage.Level.DEBUG)
    _, threshold = cv2.threshold(
        eroded,
        param.threshold,
        255,
        cv2.THRESH_BINARY,
    )
    debug.image(threshold, DebugImage.Level.DEBUG)
    # On récupère le contour le plus grand.
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    contour_max = max(contours, key=cv2.contourArea)
    image2222 = cv2.drawContours(
        cv2ext.convertion_en_couleur(image), contours, -1, (0, 0, 255), 3
    )
    image2222 = cv2.drawContours(image2222, [contour_max], 0, (0, 255, 0), 3)
    debug.image(image2222, DebugImage.Level.DEBUG)

    # On garde le rectangle le plus grand.
    rect = cv2ext.get_rectangle_from_contour_hough_lines(
        param.hough_lines, contour_max, image, debug
    )

    if rect is None:
        return None

    debug.image_lazy(
        lambda: cv2.drawContours(image2222, [rect], -1, (255, 0, 0), 3),
        DebugImage.Level.DEBUG,
    )

    # Si on n'a pas de rectangle, on essaie de trouver le contour de la
    # page avec les traits horizontaux et verticaux.
    if not compute.is_contour_rectangle(
        rect, param.pourcentage_ecart_rectangle
    ):
        return None

    return rect


@inc_debug
def found_data_try2_find_edges(
    image: np.ndarray,
    param: FoundDataTry2Parameters,
    debug: DebugImage,
) -> List[np.ndarray]:
    blurimg = cv2ext.force_image_to_be_grayscale(image, param.blur_size)

    liste_lines = []
    for i in range(2):
        if i == 0:
            threshold_param_i = param.threshold_gray
            canny_param_i = param.canny_gray
            hough_lines_param_i = param.hough_lines_gray
            morpho_mode1 = cv2.MORPH_OPEN
            morpho_mode2 = cv2.MORPH_CLOSE

            blurimg2 = blurimg
        else:
            threshold_param_i = param.threshold_histogram
            canny_param_i = param.canny_histogram
            hough_lines_param_i = param.hough_lines_histogram
            morpho_mode1 = cv2.MORPH_CLOSE
            morpho_mode2 = cv2.MORPH_OPEN

            blurimg_bc = cv2ext.apply_brightness_contrast(blurimg, -96, 64)
            debug.image(blurimg_bc, DebugImage.Level.DEBUG)
            blurimg2 = cv2.equalizeHist(blurimg_bc)

        debug.image(blurimg2, DebugImage.Level.DEBUG)

        _, threshold = cv2.threshold(
            blurimg2,
            threshold_param_i,
            255,
            cv2.THRESH_BINARY,
        )

        debug.image(threshold, DebugImage.Level.DEBUG)

        morpho1 = cv2.morphologyEx(
            threshold,
            morpho_mode1,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, param.kernel_morpho_size
            ),
        )
        debug.image(morpho1, DebugImage.Level.DEBUG)
        morpho2 = cv2.morphologyEx(
            morpho1,
            morpho_mode2,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, param.kernel_morpho_size
            ),
        )
        debug.image(morpho2, DebugImage.Level.DEBUG)
        canny = cv2.Canny(
            morpho2,
            canny_param_i.minimum,
            canny_param_i.maximum,
            apertureSize=canny_param_i.aperture_size,
        )
        debug.image(canny, DebugImage.Level.DEBUG)
        lines_i = cv2.HoughLinesP(
            canny,
            hough_lines_param_i.delta_rho,
            hough_lines_param_i.delta_tetha.get_rad(),
            hough_lines_param_i.threshold,
            minLineLength=hough_lines_param_i.min_line_length,
            maxLineGap=hough_lines_param_i.max_line_gap,
        )
        if lines_i is not None:
            liste_lines.extend(lines_i)

        def image_with_lines() -> np.ndarray:
            retval = cv2ext.convertion_en_couleur(image)

            if lines_i is None:
                return retval

            for line in lines_i:
                for point1_x, point1_y, point2_x, point2_y in line:
                    cv2.line(
                        retval,
                        (point1_x, point1_y),
                        (point2_x, point2_y),
                        (0, 0, 255),
                        1,
                    )
            return retval

        debug.image_lazy(image_with_lines, DebugImage.Level.INFO)

    height, width = cv2ext.get_hw(image)
    liste_lines.append(np.array([[0, 0, 0, height - 1]], dtype=int))
    liste_lines.append(np.array([[0, 0, width - 1, 0]], dtype=int))
    liste_lines.append(
        np.array([[width - 1, 0, width - 1, height - 1]], dtype=int)
    )
    liste_lines.append(
        np.array([[0, height - 1, width - 1, height - 1]], dtype=int)
    )
    return liste_lines


def found_data_try2_filter_edges(
    liste_lines: List[np.ndarray], images_mask: np.ndarray
) -> Tuple[
    List[Tuple[Tuple[int, int], Tuple[int, int]]],
    List[Tuple[Tuple[int, int], Tuple[int, int]]],
]:
    """Edges must be vertical or horizontal and must be not cross images."""
    lines_vertical_angle = []
    lines_horizontal_angle = []
    delta_angle = Angle.deg(3)
    for line in liste_lines:
        point1_x, point1_y, point2_x, point2_y = line[0]
        angle = compute.get_angle_0_180(
            (point1_x, point1_y), (point2_x, point2_y)
        )
        if Angle.deg(90) - delta_angle <= angle <= Angle.deg(90) + delta_angle:
            angle, posx = compute.get_angle_0_180_posx_safe(
                (point1_x, point1_y), (point2_x, point2_y)
            )
            image_line = np.zeros(images_mask.shape, np.uint8)
            cv2.line(
                image_line,
                (posx, 0),
                compute.get_bottom_point_from_alpha_posx(
                    angle, posx, images_mask.shape[0]
                ),
                (255, 255, 255),
                1,
            )
            image_line = cv2.bitwise_and(images_mask, image_line)
            if cv2.countNonZero(image_line) == 0:
                lines_vertical_angle.append(
                    ((point1_x, point1_y), (point2_x, point2_y))
                )
        if angle <= delta_angle or angle > Angle.deg(180) - delta_angle:
            angle, posy = compute.get_angle_0_180_posy_safe(
                (point1_x, point1_y), (point2_x, point2_y)
            )
            image_line = np.zeros(images_mask.shape, np.uint8)
            cv2.line(
                image_line,
                (0, posy),
                compute.get_right_point_from_alpha_posy(
                    angle, posy, images_mask.shape[1]
                ),
                (255, 255, 255),
                1,
            )
            image_line = cv2.bitwise_and(images_mask, image_line)
            if cv2.countNonZero(image_line) == 0:
                lines_horizontal_angle.append(
                    ((point1_x, point1_y), (point2_x, point2_y))
                )
    return lines_vertical_angle, lines_horizontal_angle


def found_data_try2_remove_duplicated_edges(
    lines_vertical_angle: List[Tuple[Tuple[int, int], Tuple[int, int]]],
    lines_horizontal_angle: List[Tuple[Tuple[int, int], Tuple[int, int]]],
) -> Tuple[
    List[Tuple[Tuple[int, int], Tuple[int, int]]],
    List[Tuple[Tuple[int, int], Tuple[int, int]]],
]:
    histogram_vertical: Dict[int, int] = dict()
    histogram_horizontal: Dict[int, int] = dict()
    histogram_vertical_points: Dict[
        int, Tuple[Tuple[int, int], Tuple[int, int]]
    ] = dict()
    histogram_horizontal_points: Dict[
        int, Tuple[Tuple[int, int], Tuple[int, int]]
    ] = dict()

    for line in lines_vertical_angle:
        pt1, pt2 = line
        _, posx = compute.get_angle_0_180_posx_safe(pt1, pt2)
        histogram_vertical[posx] = histogram_vertical.get(posx, 0) + 1
        histogram_vertical_points[posx] = line
    for line in lines_horizontal_angle:
        pt1, pt2 = line
        _, posy = compute.get_angle_0_180_posy_safe(pt1, pt2)
        histogram_horizontal[posy] = histogram_horizontal.get(posy, 0) + 1
        histogram_horizontal_points[posy] = line

    histogram_vertical_arr = np.zeros(max(histogram_vertical.keys()) + 1)
    histogram_horizontal_arr = np.zeros(max(histogram_horizontal.keys()) + 1)
    for key, value in histogram_vertical.items():
        histogram_vertical_arr[key] = value
    for key, value in histogram_horizontal.items():
        histogram_horizontal_arr[key] = value

    v_smooth = cv2.GaussianBlur(
        histogram_vertical_arr, (9, 9), 9, 9, cv2.BORDER_REPLICATE
    )
    h_smooth = cv2.GaussianBlur(
        histogram_horizontal_arr, (9, 9), 9, 9, cv2.BORDER_REPLICATE
    )

    lines_vertical_angle_keep: List[
        Tuple[Tuple[int, int], Tuple[int, int]]
    ] = []
    lines_horizontal_angle_keep: List[
        Tuple[Tuple[int, int], Tuple[int, int]]
    ] = []

    lines_vertical_angle_keep = compute.get_top_histogram(
        v_smooth, histogram_vertical_points
    )
    lines_horizontal_angle_keep = compute.get_top_histogram(
        h_smooth, histogram_horizontal_points
    )

    return lines_vertical_angle_keep, lines_horizontal_angle_keep


def found_data_try2_is_contour_around_images(
    zone: Tuple[int, int, int, int],
    lines_vertical_angle: List[Tuple[Tuple[int, int], Tuple[int, int]]],
    lines_horizontal_angle: List[Tuple[Tuple[int, int], Tuple[int, int]]],
    images_mask: np.ndarray,
) -> Optional[np.ndarray]:
    cnti = compute.convert_line_to_contour(
        lines_vertical_angle[zone[0]],
        lines_vertical_angle[zone[1]],
        lines_horizontal_angle[zone[2]],
        lines_horizontal_angle[zone[3]],
    )
    mask = np.zeros(images_mask.shape, np.uint8)
    mask = cv2.drawContours(mask, [cnti], -1, (255, 255, 255), -1)
    mask = cv2.bitwise_and(images_mask, mask)

    difference = cv2.subtract(images_mask, mask)

    if cv2.countNonZero(difference) == 0:
        return cnti
    return None


def found_data_try2_find_smallest_rectangular_with_all_images_inside(
    lines_vertical_angle: List[Tuple[Tuple[int, int], Tuple[int, int]]],
    lines_horizontal_angle: List[Tuple[Tuple[int, int], Tuple[int, int]]],
    images_mask: np.ndarray,
) -> np.ndarray:
    # Keep the smallest rectangle that have inside all images.
    flag_v_min: List[bool] = []
    for v_i in range(len(lines_vertical_angle)):
        v_i_min = v_i
        v_i_max = len(lines_vertical_angle) - 1
        h_i_min = 0
        h_i_max = len(lines_horizontal_angle) - 1

        cnti = found_data_try2_is_contour_around_images(
            (v_i_min, v_i_max, h_i_min, h_i_max),
            lines_vertical_angle,
            lines_horizontal_angle,
            images_mask,
        )
        flag_v_min.append(cnti is not None)
    flag_v_max: List[bool] = []
    for v_i in range(len(lines_vertical_angle) - 1, -1, -1):
        v_i_min = 0
        v_i_max = v_i
        h_i_min = 0
        h_i_max = len(lines_horizontal_angle) - 1

        cnti = found_data_try2_is_contour_around_images(
            (v_i_min, v_i_max, h_i_min, h_i_max),
            lines_vertical_angle,
            lines_horizontal_angle,
            images_mask,
        )
        flag_v_max.insert(0, cnti is not None)
    flag_h_min: List[bool] = []
    for h_i in range(len(lines_horizontal_angle)):
        v_i_min = 0
        v_i_max = len(lines_vertical_angle) - 1
        h_i_min = h_i
        h_i_max = len(lines_horizontal_angle) - 1

        cnti = found_data_try2_is_contour_around_images(
            (v_i_min, v_i_max, h_i_min, h_i_max),
            lines_vertical_angle,
            lines_horizontal_angle,
            images_mask,
        )
        flag_h_min.append(cnti is not None)
    flag_h_max: List[bool] = []
    for h_i in range(len(lines_horizontal_angle) - 1, -1, -1):
        v_i_min = 0
        v_i_max = len(lines_vertical_angle) - 1
        h_i_min = 0
        h_i_max = h_i

        cnti = found_data_try2_is_contour_around_images(
            (v_i_min, v_i_max, h_i_min, h_i_max),
            lines_vertical_angle,
            lines_horizontal_angle,
            images_mask,
        )
        flag_h_max.insert(0, cnti is not None)

    return cv2ext.bounding_rectangle(
        cv2ext.get_hw(images_mask),
        (lines_vertical_angle, lines_horizontal_angle),
        (flag_v_min, flag_v_max, flag_h_min, flag_h_max),
    )


@inc_debug
def found_data_try2(
    image: np.ndarray,
    param: FoundDataTry2Parameters,
    page_angle: Angle,
    debug: DebugImage,
) -> Optional[np.ndarray]:
    liste_lines = found_data_try2_find_edges(image, param, debug)

    images_mask = page.find_images.find_images(
        image,
        param.find_images,
        page_angle,
        debug,
    )

    if np.all(images_mask == 0):
        return None

    (
        lines_vertical_angle,
        lines_horizontal_angle,
    ) = found_data_try2_filter_edges(liste_lines, images_mask)

    (
        lines_vertical_angle,
        lines_horizontal_angle,
    ) = found_data_try2_remove_duplicated_edges(
        lines_vertical_angle, lines_horizontal_angle
    )

    lines_vertical_angle.sort(
        key=lambda x: compute.get_angle_0_180_posx_safe(x[0], x[1])[1]
    )
    lines_horizontal_angle.sort(
        key=lambda x: compute.get_angle_0_180_posy_safe(x[0], x[1])[1]
    )

    return found_data_try2_find_smallest_rectangular_with_all_images_inside(
        lines_vertical_angle, lines_horizontal_angle, images_mask
    )


@inc_debug
def crop_around_page(
    image: np.ndarray,
    parameters: CropAroundDataInPageParameters,
    page_angle: float,
    debug: DebugImage,
) -> Tuple[int, int, int, int]:
    debug.image(image, DebugImage.Level.DEBUG)

    rect = page.crop.found_data_try1(image, parameters.found_data_try1, debug)

    if rect is None:
        rect = page.crop.found_data_try2(
            image, parameters.found_data_try2, page_angle, debug
        )

    if rect is None:
        height, width = cv2ext.get_hw(image)
        return (0, width - 1, 0, height - 1)

    debug.image_lazy(
        lambda: cv2.drawContours(
            cv2ext.convertion_en_couleur(image), [rect], 0, (0, 0, 255), 3
        ),
        DebugImage.Level.DEBUG,
    )

    x_crop1 = [rect[0, 0, 0], rect[1, 0, 0], rect[2, 0, 0], rect[3, 0, 0]]
    y_crop1 = [rect[0, 0, 1], rect[1, 0, 1], rect[2, 0, 1], rect[3, 0, 1]]
    x_crop1.sort()
    y_crop1.sort()

    return (
        compute.clamp(x_crop1[0], 0, len(image[0]) - 1),
        compute.clamp(x_crop1[3], 0, len(image[0]) - 1),
        compute.clamp(y_crop1[0], 0, len(image) - 1),
        compute.clamp(y_crop1[3], 0, len(image) - 1),
    )


@inc_debug
def crop_around_data(
    page_gauche_0: np.ndarray,
    parameters: CropAroundDataInPageParameters,
    debug: DebugImage,
) -> Optional[Tuple[int, int, int, int]]:
    # On enlève les bordures noirs sur le bord des pages.
    imgh, imgw = cv2ext.get_hw(page_gauche_0)
    min_x, min_y = imgw, imgh
    max_x = max_y = 0

    gray = cv2ext.convertion_en_niveau_de_gris(page_gauche_0)

    dilated = cv2ext.erode_and_dilate(
        gray, parameters.dilate_size, parameters.dilate_size[0]
    )
    debug.image(dilated, DebugImage.Level.DEBUG)

    _, threshold = cv2.threshold(
        dilated,
        parameters.threshold2,
        255,
        cv2.THRESH_BINARY,
    )
    debug.image(threshold, DebugImage.Level.DEBUG)
    threshold2 = cv2.copyMakeBorder(
        threshold,
        1,
        1,
        1,
        1,
        cv2.BORDER_CONSTANT,
        value=[255],
    )
    contours, hierarchy = cv2.findContours(
        threshold2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = cv2ext.remove_border_in_contours(contours, 1, threshold)
    big_rectangle = parameters.get_big_rectangle(imgw, imgh)
    small_rectangle = parameters.get_small_rectangle(imgw, imgh)
    image2222 = cv2ext.convertion_en_couleur(page_gauche_0)
    image2222 = cv2.rectangle(
        image2222,
        big_rectangle[0],
        big_rectangle[1],
        (255, 0, 0),
        1,
    )
    image2222 = cv2.rectangle(
        image2222,
        small_rectangle[0],
        small_rectangle[1],
        (255, 0, 0),
        1,
    )
    ncontour_good_size = False
    first_cnt_all = int(cv2.contourArea(contours[0])) == (imgh - 1) * (
        imgw - 1
    )

    small_border = 255 * np.ones((imgh, imgw), dtype=np.uint8)
    small_border = cv2.rectangle(
        small_border,
        small_rectangle[0],
        small_rectangle[1],
        0,
        -1,
    )
    big_border = 255 * np.ones((imgh, imgw), dtype=np.uint8)
    big_border = cv2.rectangle(
        big_border,
        big_rectangle[0],
        big_rectangle[1],
        0,
        -1,
    )

    def is_border(contour: np.ndarray) -> bool:
        border_gray_inv = cv2.bitwise_not(gray)
        border_threshold_inv = cv2.bitwise_not(threshold)
        border_mask = np.zeros((imgh, imgw), dtype=np.uint8)
        cv2.drawContours(border_mask, [contour], -1, 255, -1)
        border_threshold = cv2.bitwise_and(
            border_threshold_inv, border_threshold_inv, mask=border_mask
        )
        border_gray = cv2.bitwise_not(
            cv2.bitwise_and(border_gray_inv, border_gray_inv, mask=border_mask)
        )
        cnt_in_small = cv2.bitwise_and(small_border, border_threshold)
        cnt_in_big = cv2.bitwise_and(big_border, border_threshold)

        all_in_border = cv2.countNonZero(
            cnt_in_small
        ) != 0 and cv2.countNonZero(cnt_in_big) == cv2.countNonZero(
            border_threshold
        )

        if not all_in_border:
            return False

        ptx, pty, ptw, pth = cv2.boundingRect(contour)
        image_text = border_gray[pty : pty + pth, ptx : ptx + ptw]
        return not ocr.is_text(image_text)

    contours_listered = filter(
        lambda x: parameters.contour_area_min * imgh * imgw
        < cv2.contourArea(x[0])
        < parameters.contour_area_max * imgh * imgw
        and (
            (x[1][3] == -1 and not first_cnt_all)
            or (x[1][3] == 0 and first_cnt_all)
        )
        and not is_border(x[0]),
        zip(contours, hierarchy[0]),
    )
    for cnt, _ in contours_listered:
        (point_x, point_y, width, height) = cv2.boundingRect(cnt)
        cv2.drawContours(image2222, [cnt], -1, (0, 0, 255), 3)
        min_x = min(point_x, min_x)
        max_x = max(point_x + width, max_x)
        min_y = min(point_y, min_y)
        max_y = max(point_y + height, max_y)
        ncontour_good_size = True

    debug.image(image2222, DebugImage.Level.DEBUG)

    if not ncontour_good_size:
        return None

    return (min_x, max_x, min_y, max_y)
