from typing import Any, Optional, Tuple, List, Dict, Union
import types

import numpy as np
import cv2

from parameters import ErodeParameters, CannyParameters, HoughLinesParameters
from page.find_images import FindImageParameters
import page.find_images
import cv2ext
import compute


class FoundDataTry1Parameters:
    class Impl(types.SimpleNamespace):
        erode: ErodeParameters = ErodeParameters((9, 9), 1)
        threshold: int = 240
        pourcentage_ecart_rectangle: float = 10.0

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


class FoundDataTry2Parameters:
    class Impl(types.SimpleNamespace):
        blur_size: Tuple[int, int] = (10, 10)
        threshold_gray: int = 200
        kernel_morpho_size: Tuple[int, int] = (10, 10)
        canny_gray: CannyParameters = CannyParameters(25, 255, 5)
        hough_lines_gray: HoughLinesParameters = HoughLinesParameters(
            1, np.pi / (180 * 20), 30, 100, 30
        )
        threshold_histogram: int = 15
        canny_histogram: CannyParameters = CannyParameters(25, 255, 5)
        hough_lines_histogram: HoughLinesParameters = HoughLinesParameters(
            1, np.pi / (180 * 20), 30, 100, 30
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
    class Impl(types.SimpleNamespace):
        found_data_try1: FoundDataTry1Parameters = FoundDataTry1Parameters()
        found_data_try2: FoundDataTry2Parameters = FoundDataTry2Parameters()
        dilate_size: Tuple[int, int] = (2, 2)
        threshold2: int = 200
        contour_area_min: float = 0.002 * 0.002
        contour_area_max: float = 0.5 * 0.5
        border: int = 10

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

    def init_default_values(
        self,
        key: str,
        value: Union[int, float, Tuple[int, int]],
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


def found_data_try1(
    image: Any,
    n_page: int,
    param: FoundDataTry1Parameters,
    enable_debug: Optional[str] = None,
) -> Optional[Any]:
    gray = cv2ext.convertion_en_niveau_de_gris(image)
    eroded = cv2.erode(
        gray,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, param.erode.size),
        iterations=param.erode.iterations,
    )
    if enable_debug is not None:
        cv2.imwrite(enable_debug + "_" + str(n_page) + "_2.png", eroded)
    _, threshold = cv2.threshold(
        eroded,
        param.threshold,
        255,
        cv2.THRESH_BINARY,
    )
    if enable_debug is not None:
        cv2.imwrite(enable_debug + "_" + str(n_page) + "_3.png", threshold)
    # On récupère le contour le plus grand.
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    contour_max = max(contours, key=cv2.contourArea)
    if enable_debug is not None:
        image2222 = cv2.drawContours(
            cv2ext.convertion_en_couleur(image), contours, -1, (0, 0, 255), 3
        )
        image2222 = cv2.drawContours(
            image2222, [contour_max], 0, (0, 255, 0), 3
        )
        cv2.imwrite(enable_debug + "_" + str(n_page) + "_4.png", image2222)

    # On garde le rectangle le plus grand.
    rect = cv2ext.get_polygon_from_contour(contour_max, 4)

    if enable_debug is not None:
        image22223 = cv2.drawContours(image2222, [rect], -1, (255, 0, 0), 3)
        cv2.imwrite(enable_debug + "_" + str(n_page) + "_5.png", image22223)
    # Si on n'a pas de rectangle, on essaie de trouver le contour de la
    # page avec les traits horizontaux et verticaux.
    if not compute.is_contour_rectangle(
        rect, param.pourcentage_ecart_rectangle
    ):
        return None

    return rect


def found_data_try2_find_edges(
    image: Any,
    n_page: int,
    param: FoundDataTry2Parameters,
    enable_debug: Optional[str] = None,
) -> List[Any]:
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

            blurimg2 = cv2.equalizeHist(blurimg)

        cv2ext.write_image_if(
            blurimg2,
            enable_debug,
            "_" + str(n_page) + "_" + str(i) + "_5a.png",
        )

        _, threshold = cv2.threshold(
            blurimg2,
            threshold_param_i,
            255,
            cv2.THRESH_BINARY,
        )

        cv2ext.write_image_if(
            threshold,
            enable_debug,
            "_" + str(n_page) + "_" + str(i) + "_5b.png",
        )

        morpho1 = cv2.morphologyEx(
            threshold,
            morpho_mode1,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, param.kernel_morpho_size
            ),
        )
        cv2ext.write_image_if(
            morpho1,
            enable_debug,
            "_" + str(n_page) + "_" + str(i) + "_5b1.png",
        )
        morpho2 = cv2.morphologyEx(
            morpho1,
            morpho_mode2,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, param.kernel_morpho_size
            ),
        )
        cv2ext.write_image_if(
            morpho2,
            enable_debug,
            "_" + str(n_page) + "_" + str(i) + "_5b2.png",
        )
        canny = cv2.Canny(
            morpho2,
            canny_param_i.minimum,
            canny_param_i.maximum,
            apertureSize=canny_param_i.aperture_size,
        )
        cv2ext.write_image_if(
            canny, enable_debug, "_" + str(n_page) + "_" + str(i) + "_5c.png"
        )
        lines_i = cv2.HoughLinesP(
            canny,
            hough_lines_param_i.delta_rho,
            hough_lines_param_i.delta_tetha,
            hough_lines_param_i.threshold,
            minLineLength=hough_lines_param_i.min_line_length,
            maxLineGap=hough_lines_param_i.max_line_gap,
        )
        liste_lines.extend(lines_i)
        if enable_debug is not None:
            image_with_lines = cv2ext.convertion_en_couleur(image)
            for line in lines_i:
                for point1_x, point1_y, point2_x, point2_y in line:
                    cv2.line(
                        image_with_lines,
                        (point1_x, point1_y),
                        (point2_x, point2_y),
                        (0, 0, 255),
                        1,
                    )
            cv2.imwrite(
                enable_debug + "_" + str(n_page) + "_" + str(i) + "_5d.png",
                image_with_lines,
            )
    return liste_lines


def found_data_try2_filter_edges(
    liste_lines: List[Any], images_mask: Any
) -> Tuple[
    List[Tuple[Tuple[int, int], Tuple[int, int]]],
    List[Tuple[Tuple[int, int], Tuple[int, int]]],
]:
    """Edges must be vertical or horizontal and must be not cross images."""
    lines_vertical_angle = []
    lines_horizontal_angle = []
    delta_angle = 3
    for line in liste_lines:
        point1_x, point1_y, point2_x, point2_y = line[0]
        angle = compute.get_angle_0_180(
            (point1_x, point1_y), (point2_x, point2_y)
        )
        if 90 - delta_angle <= angle <= 90 + delta_angle:
            angle, posx = compute.get_angle_0_180_posx(
                (point1_x, point1_y), (point2_x, point2_y)
            )
            if posx is None:
                raise Exception("Line can't be horizontal")
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
        if angle <= delta_angle or angle > 180 - delta_angle:
            angle, posy = compute.get_alpha_posy(
                (point1_x, point1_y), (point2_x, point2_y)
            )
            if posy is None:
                raise Exception("Line can't be vertical")
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
        _, posx = compute.get_angle_0_180_posx(pt1, pt2)
        if posx is None:
            raise Exception("Ligne can't be horizontal")
        histogram_vertical[posx] = histogram_vertical.get(posx, 0) + 1
        histogram_vertical_points[posx] = line
    for line in lines_horizontal_angle:
        pt1, pt2 = line
        _, posy = compute.get_alpha_posy(pt1, pt2)
        if posy is None:
            raise Exception("Ligne can't be vertical")
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
    images_mask: Any,
) -> Optional[Any]:
    point1_x, point1_y = compute.line_intersection(
        lines_vertical_angle[zone[0]], lines_horizontal_angle[zone[2]]
    )
    point2_x, point2_y = compute.line_intersection(
        lines_vertical_angle[zone[0]], lines_horizontal_angle[zone[3]]
    )
    point3_x, point3_y = compute.line_intersection(
        lines_vertical_angle[zone[1]], lines_horizontal_angle[zone[2]]
    )
    point4_x, point4_y = compute.line_intersection(
        lines_vertical_angle[zone[1]], lines_horizontal_angle[zone[3]]
    )

    xmoy = (point1_x + point2_x + point3_x + point4_x) // 4
    ymoy = (point1_y + point2_y + point3_y + point4_y) // 4
    list_of_points = [
        [point1_x, point1_y],
        [point2_x, point2_y],
        [point3_x, point3_y],
        [point4_x, point4_y],
    ]

    list_of_points.sort(
        key=lambda x: compute.get_angle__180_180((xmoy, ymoy), (x[0], x[1]))
    )
    cnti = np.asarray(list_of_points)
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
    images_mask: Any,
) -> Optional[Any]:
    cnt = []

    v_min = 0
    v_max = len(lines_vertical_angle) - 1
    h_min = 0
    h_max = len(lines_horizontal_angle) - 1

    # Keep the smallest rectangle that have inside all images.
    for v_i in range(len(lines_vertical_angle)):
        v_i_min = v_i
        v_i_max = v_max
        h_i_min = h_min
        h_i_max = h_max

        cnti = found_data_try2_is_contour_around_images(
            (v_i_min, v_i_max, h_i_min, h_i_max),
            lines_vertical_angle,
            lines_horizontal_angle,
            images_mask,
        )
        if cnti is None:
            break
        v_min = v_i
        cnt = cnti
    for v_i in range(len(lines_vertical_angle) - 1, v_min, -1):
        v_i_min = v_min
        v_i_max = v_i
        h_i_min = h_min
        h_i_max = h_max

        cnti = found_data_try2_is_contour_around_images(
            (v_i_min, v_i_max, h_i_min, h_i_max),
            lines_vertical_angle,
            lines_horizontal_angle,
            images_mask,
        )
        if cnti is None:
            break
        v_i_max = v_i
        cnt = cnti
    for h_i in range(len(lines_horizontal_angle)):
        v_i_min = v_min
        v_i_max = v_max
        h_i_min = h_i
        h_i_max = h_max

        cnti = found_data_try2_is_contour_around_images(
            (v_i_min, v_i_max, h_i_min, h_i_max),
            lines_vertical_angle,
            lines_horizontal_angle,
            images_mask,
        )
        if cnti is None:
            break
        h_min = h_i
        cnt = cnti

    for h_i in range(len(lines_horizontal_angle) - 1, h_min, -1):
        v_i_min = v_min
        v_i_max = v_max
        h_i_min = h_min
        h_i_max = h_i

        cnti = found_data_try2_is_contour_around_images(
            (v_i_min, v_i_max, h_i_min, h_i_max),
            lines_vertical_angle,
            lines_horizontal_angle,
            images_mask,
        )
        if cnti is None:
            break
        h_max = h_i
        cnt = cnti

    if len(cnt) == 0:
        return None

    return cv2ext.get_polygon_from_contour(cnt, 4)


def found_data_try2(
    image: Any,
    n_page: int,
    param: FoundDataTry2Parameters,
    enable_debug: Optional[str] = None,
) -> Optional[Any]:
    liste_lines = found_data_try2_find_edges(
        image, n_page, param, enable_debug
    )

    images_mask = page.find_images.find_images(
        image,
        param.find_images,
        compute.optional_concat(enable_debug, "_A_crop_" + str(n_page)),
    )

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

    lines_vertical_angle.sort(key=compute.sort_edges_by_posx)
    lines_horizontal_angle.sort(key=compute.sort_edges_by_posy)

    return found_data_try2_find_smallest_rectangular_with_all_images_inside(
        lines_vertical_angle, lines_horizontal_angle, images_mask
    )


def crop_around_page(
    image: Any,
    n_page: int,
    parameters: CropAroundDataInPageParameters,
    enable_debug: Optional[str] = None,
) -> Tuple[int, int, int, int]:
    if enable_debug is not None:
        cv2.imwrite(enable_debug + "_" + str(n_page) + "_1.png", image)

    rect = page.crop.found_data_try1(
        image, n_page, parameters.found_data_try1, enable_debug
    )

    if rect is None:
        rect = page.crop.found_data_try2(
            image, n_page, parameters.found_data_try2, enable_debug
        )

    if rect is None:
        raise Exception("Failed to found contour of the page.")
    x_crop1 = [rect[0, 0, 0], rect[1, 0, 0], rect[2, 0, 0], rect[3, 0, 0]]
    y_crop1 = [rect[0, 0, 1], rect[1, 0, 1], rect[2, 0, 1], rect[3, 0, 1]]
    x_crop1.sort()
    y_crop1.sort()

    return (x_crop1[1], x_crop1[2], y_crop1[1], y_crop1[2])


def crop_around_data(
    page_gauche_0: Any,
    n_page: int,
    parameters: CropAroundDataInPageParameters,
    enable_debug: Optional[str] = None,
) -> Optional[Any]:
    # On enlève les bordures noirs sur le bord des pages.
    imgh, imgw = cv2ext.get_hw(page_gauche_0)
    min_x, min_y = imgw, imgh
    max_x = max_y = 0

    gray = cv2ext.convertion_en_niveau_de_gris(page_gauche_0)

    dilated = cv2.dilate(
        gray,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, parameters.dilate_size),
    )
    if enable_debug is not None:
        cv2.imwrite(enable_debug + "_" + str(n_page) + "_7.png", dilated)

    _, threshold = cv2.threshold(
        dilated,
        parameters.threshold2,
        255,
        cv2.THRESH_BINARY,
    )
    if enable_debug is not None:
        cv2.imwrite(enable_debug + "_" + str(n_page) + "_8.png", threshold)
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    image2222 = page_gauche_0.copy()
    ncontour_good_size = 0
    contours_listered = filter(
        lambda x: parameters.contour_area_min * imgh * imgw
        < cv2.contourArea(x)
        < parameters.contour_area_max * imgh * imgw,
        contours,
    )
    for cnt in contours_listered:
        (point_x, point_y, width, height) = cv2.boundingRect(cnt)
        cv2.drawContours(image2222, [cnt], -1, (0, 0, 255), 3)
        min_x = min(point_x, min_x)
        max_x = max(point_x + width, max_x)
        min_y = min(point_y, min_y)
        max_y = max(point_y + height, max_y)
        ncontour_good_size = ncontour_good_size + 1

    if enable_debug is not None:
        cv2.imwrite(enable_debug + "_" + str(n_page) + "_9.png", image2222)

    if ncontour_good_size == 0:
        return None

    return (min_x, max_x, min_y, max_y)
