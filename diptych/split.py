from itertools import combinations
import types
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import scipy
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import binary_erosion, generate_binary_structure

from . import compute, cv2ext
from .angle import Angle
from .debug_image import DebugImage, inc_debug
from .find_images import (
    FindImageParameters,
    remove_points_inside_images_in_contours,
)
from .parameters import CannyParameters, ErodeParameters, HoughLinesParameters


class FoundSplitLineWithLineParameters:
    class Impl(types.SimpleNamespace):
        blur_size: Tuple[int, int]
        erode: ErodeParameters
        canny: CannyParameters
        hough_lines: HoughLinesParameters
        limit_rho: int
        limit_tetha: Angle

    def __init__(  # pylint: disable=too-many-arguments
        self,
        blur_size: Tuple[int, int],
        erode: ErodeParameters,
        canny: CannyParameters,
        hough_lines: HoughLinesParameters,
        limit_rho: int,
        limit_tetha: Angle,
    ) -> None:
        self.__param = FoundSplitLineWithLineParameters.Impl(
            blur_size=blur_size,
            erode=erode,
            canny=canny,
            hough_lines=hough_lines,
            limit_rho=limit_rho,
            limit_tetha=limit_tetha,
        )

    @property
    def blur_size(self) -> Tuple[int, int]:
        return self.__param.blur_size

    @blur_size.setter
    def blur_size(self, val: Tuple[int, int]) -> None:
        self.__param.blur_size = val

    @property
    def erode(self) -> ErodeParameters:
        return self.__param.erode

    @property
    def canny(self) -> CannyParameters:
        return self.__param.canny

    @property
    def hough_lines(
        self,
    ) -> HoughLinesParameters:
        return self.__param.hough_lines

    @property
    def limit_rho(self) -> int:
        return self.__param.limit_rho

    @limit_rho.setter
    def limit_rho(self, val: int) -> None:
        self.__param.limit_rho = val

    @property
    def limit_tetha(self) -> Angle:
        return self.__param.limit_tetha

    @limit_tetha.setter
    def limit_tetha(self, val: Angle) -> None:
        self.__param.limit_tetha = val


class FindCandidatesSplitLineWithWaveParameters:
    class Impl(types.SimpleNamespace):
        rapport_rect1_rect2: float

    def __init__(
        self,
        rapport_rect1_rect2: float,
    ) -> None:
        self.__param = FindCandidatesSplitLineWithWaveParameters.Impl(
            rapport_rect1_rect2=rapport_rect1_rect2,
        )

    @property
    def rapport_rect1_rect2(
        self,
    ) -> float:
        return self.__param.rapport_rect1_rect2

    @rapport_rect1_rect2.setter
    def rapport_rect1_rect2(self, val: float) -> None:
        self.__param.rapport_rect1_rect2 = val

    def init_default_values(
        self,
        key: str,
        value: Union[int, float, Tuple[int, int], Angle],
    ) -> None:
        if key == "RapportRect1Rect2" and isinstance(value, float):
            self.rapport_rect1_rect2 = value
        else:
            raise Exception("Invalid property.", key)


class FoundSplitLineWithWave:
    class Impl(types.SimpleNamespace):
        blur_size: Tuple[int, int]
        erode: ErodeParameters
        find_images: FindImageParameters
        find_candidates: FindCandidatesSplitLineWithWaveParameters

    def __init__(
        self,
        blur_size: Tuple[int, int],
        erode: ErodeParameters,
        find_images: FindImageParameters,
        find_candidates: FindCandidatesSplitLineWithWaveParameters,
    ) -> None:
        self.__param = FoundSplitLineWithWave.Impl(
            blur_size=blur_size,
            erode=erode,
            find_images=find_images,
            find_candidates=find_candidates,
        )

    @property
    def blur_size(self) -> Tuple[int, int]:
        return self.__param.blur_size

    @blur_size.setter
    def blur_size(self, val: Tuple[int, int]) -> None:
        self.__param.blur_size = val

    @property
    def erode(self) -> ErodeParameters:
        return self.__param.erode

    @property
    def find_images(self) -> FindImageParameters:
        return self.__param.find_images

    @property
    def find_candidates(
        self,
    ) -> FindCandidatesSplitLineWithWaveParameters:
        return self.__param.find_candidates


class FindCandidatesSplitLineWithLineParameters:
    class Impl(types.SimpleNamespace):
        blur_size: Tuple[int, int]
        canny: CannyParameters
        hough_lines: HoughLinesParameters
        erode: ErodeParameters
        limit_rho: int
        limit_tetha: Angle

    def __init__(  # pylint: disable=too-many-arguments
        self,
        blur_size: Tuple[int, int],
        canny: CannyParameters,
        hough_lines: HoughLinesParameters,
        erode: ErodeParameters,
        limit_rho: int,
        limit_tetha: Angle,
    ) -> None:
        self.__param = FindCandidatesSplitLineWithLineParameters.Impl(
            blur_size=blur_size,
            canny=canny,
            hough_lines=hough_lines,
            erode=erode,
            limit_rho=limit_rho,
            limit_tetha=limit_tetha,
        )

    @property
    def blur_size(
        self,
    ) -> Tuple[int, int]:
        return self.__param.blur_size

    @blur_size.setter
    def blur_size(self, val: Tuple[int, int]) -> None:
        self.__param.blur_size = val

    @property
    def canny(
        self,
    ) -> CannyParameters:
        return self.__param.canny

    @property
    def hough_lines(
        self,
    ) -> HoughLinesParameters:
        return self.__param.hough_lines

    @property
    def erode(
        self,
    ) -> ErodeParameters:
        return self.__param.erode

    @property
    def limit_rho(self) -> int:
        return self.__param.limit_rho

    @limit_rho.setter
    def limit_rho(self, val: int) -> None:
        self.__param.limit_rho = val

    @property
    def limit_tetha(
        self,
    ) -> Angle:
        return self.__param.limit_tetha

    @limit_tetha.setter
    def limit_tetha(self, val: Angle) -> None:
        self.__param.limit_tetha = val


class SplitTwoWavesParameters:
    class Impl(types.SimpleNamespace):
        erode: ErodeParameters = ErodeParameters((3, 3), 10)
        blur_size: Tuple[int, int] = (10, 10)
        canny: CannyParameters = CannyParameters(25, 255, 5)
        hough_lines: HoughLinesParameters = HoughLinesParameters(
            1, Angle.deg(1 / 20), 100, 200, 25, 0.35
        )
        delta_rho: int = 200
        delta_tetha: Angle = Angle.deg(20.0)
        find_images: FindImageParameters = FindImageParameters(
            5, (10, 10), (10, 10), (10, 10), 0.01
        )
        find_candidates: FindCandidatesSplitLineWithWaveParameters = (
            FindCandidatesSplitLineWithWaveParameters(
                1.05,
            )
        )

    def __init__(self) -> None:
        self.__param = SplitTwoWavesParameters.Impl()

    @property
    def erode(self) -> ErodeParameters:
        return self.__param.erode

    @property
    def blur_size(self) -> Tuple[int, int]:
        return self.__param.blur_size

    @blur_size.setter
    def blur_size(self, val: Tuple[int, int]) -> None:
        self.__param.blur_size = val

    @property
    def canny(self) -> CannyParameters:
        return self.__param.canny

    @property
    def hough_lines(self) -> HoughLinesParameters:
        return self.__param.hough_lines

    @property
    def delta_tetha(self) -> Angle:
        return self.__param.delta_tetha

    @delta_tetha.setter
    def delta_tetha(self, val: Angle) -> None:
        self.__param.delta_tetha = val

    @property
    def delta_rho(self) -> int:
        return self.__param.delta_rho

    @delta_rho.setter
    def delta_rho(self, val: int) -> None:
        self.__param.delta_rho = val

    @property
    def find_images(self) -> FindImageParameters:
        return self.__param.find_images

    @property
    def find_candidates(
        self,
    ) -> FindCandidatesSplitLineWithWaveParameters:
        return self.__param.find_candidates

    def init_default_values(
        self,
        key: str,
        value: Union[int, float, Tuple[int, int], Angle],
    ) -> None:
        if key.startswith("Erode"):
            self.erode.init_default_values(key[len("Erode") :], value)
        elif key == "BlurSize" and isinstance(value, tuple):
            self.blur_size = value
        elif key.startswith("Canny"):
            self.canny.init_default_values(key[len("Canny") :], value)
        elif key.startswith("HoughLines"):
            self.hough_lines.init_default_values(
                key[len("HoughLines") :], value
            )
        elif key == "DeltaRho" and isinstance(value, int):
            self.delta_rho = value
        elif key == "DeltaTetha" and isinstance(value, Angle):
            self.delta_tetha = value
        elif key.startswith("Wave"):
            self.find_candidates.init_default_values(key[len("Wave") :], value)
        else:
            raise Exception("Invalid property.", key)


def __found_candidates_split_line_with_line(
    image: np.ndarray,
    images_mask: np.ndarray,
    param: FindCandidatesSplitLineWithLineParameters,
    debug: DebugImage,
) -> List[Tuple[int, int, int, int]]:
    xxx = 7
    blurimg = cv2ext.force_image_to_be_grayscale(image, (xxx, xxx))
    debug.image(blurimg, DebugImage.Level.DEBUG)

    small = cv2.resize(
        blurimg,
        (0, 0),
        fx=param.hough_lines.scale,
        fy=param.hough_lines.scale,
    )

    debug.image(small, DebugImage.Level.DEBUG)

    images_mask_small = cv2.resize(
        images_mask,
        (0, 0),
        fx=param.hough_lines.scale,
        fy=param.hough_lines.scale,
    )

    canny = cv2.Canny(
        small,
        param.canny.minimum,
        param.canny.maximum,
        apertureSize=param.canny.aperture_size,
    )
    debug.image(canny, DebugImage.Level.DEBUG)
    canny_filtered = cv2.bitwise_and(canny, cv2.bitwise_not(images_mask_small))
    debug.image(canny_filtered, DebugImage.Level.DEBUG)

    list_lines_p = cv2.HoughLinesP(
        canny_filtered,
        param.hough_lines.delta_rho,
        param.hough_lines.delta_tetha.get_rad(),
        param.hough_lines.threshold,
        minLineLength=param.hough_lines.min_line_length,
        # Distance between two lines of text.
        # You have to hope that width spaces are smaller.
        maxLineGap=param.hough_lines.max_line_gap,
    )
    debug.image_lazy(
        lambda: cv2ext.draw_lines_from_hough_lines(
            small, list_lines_p, (0, 0, 255), 1
        ),
        DebugImage.Level.DEBUG,
    )
    lines_valid = (
        np.asarray(
            list(
                filter(
                    lambda x: compute.keep_angle_pos_closed_to_target(
                        x[0], param.limit_tetha, Angle.deg(90)
                    ),
                    list_lines_p,
                )
            )
        )
        / param.hough_lines.scale
    ).astype(np.int32)
    debug.image(
        cv2ext.draw_lines_from_hough_lines(image, lines_valid, (0, 0, 255), 1),
        DebugImage.Level.DEBUG,
    )

    return list(map(lambda p: p[0], lines_valid))


def detect_peaks(image: np.ndarray) -> np.ndarray:
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(image, footprint=neighborhood) == image
    background = image == 0
    eroded_background = binary_erosion(
        background, structure=neighborhood, border_value=1
    )
    detected_peaks = local_max ^ eroded_background

    return detected_peaks


def __loop_to_find_best_mean_angle_pos(
    histogram_posx_length: np.ndarray,
    posx_ecart: int,
    epsilon_angle: Angle,
) -> Tuple[Angle, int, List[Tuple[Angle, int, int]], int]:
    local_maximum = detect_peaks(histogram_posx_length)
    local_maximum_pos_raw = np.where(local_maximum)
    all_max_length = list(
        map(
            lambda x: (
                epsilon_angle * x[0],
                x[1] * posx_ecart,
                histogram_posx_length[x[0], x[1]],
            ),
            zip(local_maximum_pos_raw[0], local_maximum_pos_raw[1]),
        )
    )
    max_length = sorted(all_max_length, key=lambda x: x[2], reverse=True)[0][2]
    all_only_max_length = list(
        filter(lambda x: x[2] > max_length / 5, all_max_length)
    )
    _, posxs, lengths = list(zip(*all_only_max_length))
    moy_posx = (min(posxs) + max(posxs)) / 2
    restricted_posx = list(
        filter(lambda x: 0.95 * moy_posx <= x <= 1.05 * moy_posx, posxs)
    )
    if len(restricted_posx) == 0:
        moy_posx = posxs[np.argmax(lengths)]
    else:
        moy_posx = sorted(restricted_posx, key=lambda x: np.abs(x - moy_posx))[
            0
        ]

    best_angle_with_err = list(
        map(
            lambda x: (
                x[0],
                x[1],
                x[2],
                x[2]
                * (
                    1.0
                    - 2.0
                    * np.absolute(compute.norm_cdf(x[1], moy_posx, 10) - 0.5)
                ),
            ),
            all_only_max_length,
        )
    )

    return (
        compute.mean_angle(
            list(zip(*best_angle_with_err))[0],
            list(zip(*best_angle_with_err))[3],
        ),
        int(
            compute.mean_weight(
                list(zip(*best_angle_with_err))[1],
                list(zip(*best_angle_with_err))[3],
            )
        ),
        all_max_length,
        posx_ecart,
    )


def __best_candidates_split_line_with_line(
    valid_lines: List[Tuple[int, int, int, int]],
    width: int,
    height: int,
    epsilon_angle: Angle,
) -> Tuple[Angle, int, List[Tuple[Angle, int, int]], int]:
    ecart = (
        int(
            np.ceil(
                np.tan(epsilon_angle.get_rad())
                * np.linalg.norm(np.array((width, height)))
            )
        )
        + 1
    )

    histogram_size_angle = int(np.ceil(180.0 / epsilon_angle.get_deg()))
    histogram_size_posx = (width + ecart - 1) // ecart
    # https://stackoverflow.com/questions/40709519/initialize-64-by-64-numpy-of-0-0-tuples-in-python
    value = np.empty((), dtype=object)
    value[()] = []
    histogram = np.full(
        (
            histogram_size_angle,
            histogram_size_posx,
        ),
        value,
        dtype=object,
    )
    tolerance = 4
    for point1_x, point1_y, point2_x, point2_y in valid_lines:
        angle, pos = compute.get_angle_0_180_posx_safe(
            (point1_x, point1_y), (point2_x, point2_y)
        )
        angle_round = int(np.round(angle / epsilon_angle))
        pos_round = int(np.round(pos / ecart))

        min_angle_round = max(angle_round - tolerance, 0)
        min_pos_round = max(pos_round - tolerance, 0)
        max_angle_round = min(
            angle_round + tolerance, histogram_size_angle - 1
        )
        max_pos_round = min(pos_round + tolerance, histogram_size_posx - 1)
        for i in range(min_angle_round, max_angle_round + 1):
            for j in range(min_pos_round, max_pos_round + 1):
                new_list = histogram[i, j].copy()
                new_list.append(
                    (min(point1_y, point2_y), max(point1_y, point2_y))
                )
                histogram[i, j] = compute.merge_interval(new_list)

    histogram_length = np.asarray(
        (
            [
                np.asarray(
                    (
                        [
                            0 if len(y) == 0 else sum([z[1] - z[0] for z in y])
                            for y in x
                        ]
                    ),
                    dtype=np.float32,
                )
                for x in histogram
            ]
        ),
        dtype=np.float32,
    )
    histogram_length_blur = cv2.GaussianBlur(
        histogram_length,
        (11, 11),
        11,
        borderType=cv2.BORDER_REPLICATE,
    )
    histogram_length_blur_int = histogram_length_blur.astype(int)

    return __loop_to_find_best_mean_angle_pos(
        histogram_length_blur_int,
        ecart,
        epsilon_angle,
    )


@inc_debug
def found_split_line_with_line(
    image: np.ndarray,
    images_found: np.ndarray,
    param: FoundSplitLineWithLineParameters,
    debug: DebugImage,
) -> Tuple[Angle, int, List[Tuple[Angle, int, int]], int]:
    debug.image(image, DebugImage.Level.DEBUG)

    valid_lines = __found_candidates_split_line_with_line(
        image,
        images_found,
        FindCandidatesSplitLineWithLineParameters(
            param.blur_size,
            param.canny,
            param.hough_lines,
            param.erode,
            param.limit_rho,
            param.limit_tetha,
        ),
        debug,
    )

    if len(valid_lines) == 0:
        raise Exception(
            "found_split_line_with_line",
            "Failed to find candidates for the separator line.",
        )

    (
        angle_1,
        posx_1,
        histogram_length,
        ecart,
    ) = __best_candidates_split_line_with_line(
        valid_lines,
        cv2ext.get_hw(image)[1],
        cv2ext.get_hw(image)[0],
        param.hough_lines.delta_tetha,
    )

    height, _ = cv2ext.get_hw(image)
    point_1a = (posx_1, 0)
    point_1b = (
        int(posx_1 - np.tan(angle_1.get_rad() - np.pi / 2) * height),
        height - 1,
    )

    (angle_2, posx_2) = cv2ext.best_fitline(
        point_1a, point_1b, valid_lines, cv2ext.get_hw(image), 2 * ecart
    )

    point_2a = (posx_2, 0)
    point_2b = (
        int(posx_2 - np.tan(angle_2.get_rad() - np.pi / 2) * height),
        height - 1,
    )

    def image_with_lines() -> np.ndarray:
        retval = cv2ext.convertion_en_couleur(image)
        cv2.line(
            retval,
            (point_1a[0], point_1a[1]),
            (point_1b[0], point_1b[1]),
            (255, 0, 0),
            3,
        )
        cv2.line(
            retval,
            (point_2a[0], point_2a[1]),
            (point_2b[0], point_2b[1]),
            (0, 255, 0),
            3,
        )
        for line in valid_lines:
            cv2.line(
                retval,
                (line[0], line[1]),
                (line[2], line[3]),
                (0, 0, 255),
                1,
            )
        return retval

    debug.image_lazy(image_with_lines, DebugImage.Level.TOP)

    return angle_2, posx_2, histogram_length, ecart


def __found_best_split_line_with_wave_hull(
    contour: np.ndarray,
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    hull = cv2.convexHull(contour, returnPoints=False)
    defects = cv2.convexityDefects(contour, hull)
    if defects is None:
        return None
    defects_full_data = [
        (
            contour[x[0, 0]][0],
            contour[x[0, 1]][0],
            contour[x[0, 2]][0],
            x[0, 3],
            np.array(
                (
                    compute.get_perpendicular_throught_point(
                        contour[x[0, 0]][0],
                        contour[x[0, 1]][0],
                        contour[x[0, 2]][0],
                    )
                )
            ),
        )
        for x in defects
    ]
    defects_valid = list(filter(lambda x: x[3] > 2 * 256, defects_full_data))
    candidates = []

    for defect1, defect2 in combinations(defects_valid, 2):
        data = list(
            zip(
                *(
                    defect1[2],
                    defect1[4],
                    defect2[2],
                    defect2[4],
                )
            )
        )
        list_angle = [
            compute.get_angle_0_180(
                (defect1[2][0], defect1[2][1]), (defect1[4][0], defect1[4][1])
            ),
            compute.get_angle_0_180(
                (defect1[4][0], defect1[4][1]), (defect2[2][0], defect2[2][1])
            ),
            compute.get_angle_0_180(
                (defect2[2][0], defect2[2][1]), (defect2[4][0], defect2[4][1])
            ),
        ]
        mean_angle = compute.mean_angle(list_angle) % Angle.deg(180.0)
        if Angle.deg(45.0) < mean_angle < Angle.deg(135.0):
            x_s = data[1]
            y_s = data[0]
        else:
            x_s = data[0]
            y_s = data[1]

        linres = scipy.stats.linregress(x_s, y_s)
        # Don't use rvalue.
        # With data ((3104, 3506, 441, 0), (2498, 2498, 2498, 2500))
        # rvalue**2 = 0.427 but in reality, it's almost perfect.
        error = sum(
            [(linres[1] + x * linres[0] - y) ** 2 for x, y in zip(x_s, y_s)]
        )

        candidates.append(
            (
                data,
                error,
                defect1[3] + defect2[3],
                sum(
                    [
                        np.square(x.get_deg() - mean_angle.get_deg())
                        for x in list_angle
                    ]
                )
                / len(list_angle),
                defect1[4].astype(int),
                defect2[4].astype(int),
            )
        )

    keep_best_candidates = filter(
        lambda x: x[1] < 200 and x[2] > 10000 and x[3] < 2, candidates
    )
    result_sorted_by_same_angle = sorted(
        keep_best_candidates, key=lambda x: x[3]
    )

    # May fail if only one contour is found at the previous step and
    # the contour is around only one page, not two.
    if len(result_sorted_by_same_angle) == 0:
        return None

    return (
        (
            (
                result_sorted_by_same_angle[0][4][0],
                result_sorted_by_same_angle[0][4][1],
            )
        ),
        (
            (
                result_sorted_by_same_angle[0][5][0],
                result_sorted_by_same_angle[0][5][1],
            )
        ),
    )


def __found_best_split_line_with_wave_n_contours(  # noqa
    contours: List[np.ndarray],
    n_contours: int,
    image: np.ndarray,
    debug: DebugImage,
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    cnt_i = 0
    split_lines = []
    for contour_i in contours[0:n_contours]:
        cnt_i = cnt_i + 1
        polygon = cv2ext.get_polygon_from_contour(contour_i, 4)

        def img_tmp() -> np.ndarray:
            retval = cv2.drawContours(
                cv2ext.convertion_en_couleur(image),
                [polygon],
                0,
                (0, 0, 255),
                10,
            )
            for i in range(n_contours):
                cv2.drawContours(
                    retval, contours, i, (255 * (1 - i), 255 * i, 0), 10
                )
            return retval

        debug.image_lazy(img_tmp, DebugImage.Level.DEBUG)
        lines, usefull_points = cv2ext.convert_polygon_with_fitline(
            contour_i, polygon
        )
        if len(lines) == 0 and len(usefull_points) == 0:
            candidate1 = None
        else:

            def image_with_lines() -> np.ndarray:
                retval = cv2ext.convertion_en_couleur(image)
                for line in lines:
                    cv2.line(
                        retval,
                        line[0],
                        line[1],
                        (0, 0, 255),
                        5,
                    )
                for point_i in usefull_points:
                    retval[point_i[1], point_i[0]] = np.asarray(
                        (255, 0, 0), dtype=np.uint8
                    )
                return retval

            debug.image_lazy(image_with_lines, DebugImage.Level.DEBUG)

            lines_sorted_by_length = sorted(
                lines,
                key=lambda x: np.linalg.norm(
                    np.array((x[0][0], x[0][1])) - np.array((x[1][0], x[1][1]))
                ),
                reverse=True,
            )
            lines_two_longest = lines_sorted_by_length[0:2]
            lines_two_longest_with_distance = [
                (
                    x[0],
                    x[1],
                    compute.get_distance_line_point(
                        x[0],
                        x[1],
                        ((image.shape[1] / 2, image.shape[0] / 2)),
                    ),
                )
                for x in lines_two_longest
            ]
            # Keep only line that are closed to the center.
            # The line is supposed to split a page in two.
            lines_sorted_by_distance_to_center = sorted(
                filter(
                    lambda x: x[2] < min(image.shape[1], image.shape[0]) / 4,
                    lines_two_longest_with_distance,
                ),
                key=lambda x: x[2],
                reverse=False,
            )
            if len(lines_sorted_by_distance_to_center) == 0:
                candidate1 = None
            else:
                candidate1 = (
                    lines_sorted_by_distance_to_center[0][0],
                    lines_sorted_by_distance_to_center[0][1],
                )

        # Check if it's only one contour
        # because the contour is outside the two pages or
        # because only one page is detected.
        if n_contours == 1:
            candidate2 = __found_best_split_line_with_wave_hull(contour_i)

            lines_two_longest_2 = [
                (
                    x[0],
                    x[1],
                    compute.get_distance_line_point(
                        x[0],
                        x[1],
                        ((image.shape[1] / 2, image.shape[0] / 2)),
                    ),
                )
                for x in [candidate1, candidate2]
                if x is not None
            ]

            if len(lines_two_longest_2) == 0:
                return None

            lines_sorted_by_distance_to_center_2 = sorted(
                lines_two_longest_2,
                key=lambda x: x[2],
                reverse=False,
            )
            return (
                lines_sorted_by_distance_to_center_2[0][0],
                lines_sorted_by_distance_to_center_2[0][1],
            )

        if candidate1 is not None:
            split_lines.append(candidate1)

    return tuple(  # type: ignore
        map(
            lambda y: tuple(map(lambda z: int(sum(z) / len(z)), zip(*y))),
            zip(*split_lines),
        )
    )


def __found_best_split_line_with_wave(
    contour: List[np.ndarray],
    image: np.ndarray,
    eroded: np.ndarray,
    param: FindCandidatesSplitLineWithWaveParameters,
    debug: DebugImage,
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    if len(contour) == 1:
        nb_rectangle = 1
    else:
        nb_rectangle = (
            int(
                cv2.contourArea(contour[0]) / cv2.contourArea(contour[1])
                < param.rapport_rect1_rect2
            )
            + 1
        )

    def img_tmp() -> np.ndarray:
        retval = cv2.cvtColor(eroded, cv2.COLOR_GRAY2BGR)
        for i in range(nb_rectangle):
            retval = cv2.drawContours(
                retval, contour, i, (255 * (1 - i), 255 * i, 0), 10
            )
        if len(contour) > nb_rectangle:
            retval = cv2.drawContours(
                retval, contour, nb_rectangle, (0, 0, 255), 3
            )
        return retval

    debug.image_lazy(img_tmp, DebugImage.Level.DEBUG)

    return __found_best_split_line_with_wave_n_contours(
        contour, nb_rectangle, image, debug
    )


@inc_debug
def found_split_line_with_wave(
    image: np.ndarray,
    parameters: FoundSplitLineWithWave,
    page_angle: Optional[Angle],
    debug: DebugImage,
) -> Optional[Tuple[Angle, int]]:
    xxx = 7
    blurimg = cv2ext.force_image_to_be_grayscale(image, (xxx, xxx))
    debug.image(blurimg, DebugImage.Level.DEBUG)
    erode_dilate = cv2ext.erode_and_dilate(blurimg, (xxx, xxx), xxx)
    debug.image(erode_dilate, DebugImage.Level.DEBUG)
    size_border = 20
    eroded_bordered = cv2.copyMakeBorder(
        erode_dilate,
        size_border,
        size_border,
        size_border,
        size_border,
        cv2.BORDER_CONSTANT,
        value=[0],
    )
    debug.image(eroded_bordered, DebugImage.Level.DEBUG)
    dilated = cv2.dilate(
        eroded_bordered,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (xxx, xxx)),
    )
    debug.image(dilated, DebugImage.Level.DEBUG)
    thresholdi = cv2ext.threshold_from_gaussian_histogram_white(dilated)
    _, threshold1 = cv2.threshold(dilated, thresholdi, 255, cv2.THRESH_BINARY)
    debug.image(threshold1, DebugImage.Level.DEBUG)

    # On cherche tous les contours
    contours, _ = cv2.findContours(
        threshold1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )
    # pour ne garder que le plus grand. Normalement, cela doit être celui
    # qui fait le contour des pages
    # et non le tour du bord de l'image.
    # On tri les contours du plus grand au plus petit.
    # L'intérêt est de comparer la taille du premier et du deuxième
    # contour.
    # Si le bord de la page est trop en haut et en bas, plutôt que d'avoir
    # un contour qui fait les deux pages, on peut avoir deux contours qui
    # font chaque page.
    sorted_contours = sorted(
        contours,
        key=lambda x: np.maximum(cv2.contourArea(x), 1),
        reverse=True,
    )

    def img_contours() -> np.ndarray:
        retval = cv2.cvtColor(eroded_bordered, cv2.COLOR_GRAY2BGR)
        for i in range(min(2, len(sorted_contours))):
            retval = cv2.drawContours(
                retval, sorted_contours, i, (255 * (1 - i), 255 * i, 0), 10
            )
        return retval

    debug.image_lazy(img_contours, DebugImage.Level.DEBUG)

    sorted_contours = cv2ext.remove_border_in_contours(
        sorted_contours, size_border, image
    )

    cs2 = remove_points_inside_images_in_contours(
        sorted_contours,
        image,
        parameters.find_images,
        None if page_angle is None else page_angle - Angle.deg(90.0),
        debug,
    )

    if len(cs2) == 0:
        return None

    line_wave = __found_best_split_line_with_wave(
        cs2,
        image,
        erode_dilate,
        parameters.find_candidates,
        debug,
    )

    if line_wave is None:
        return None

    bottompoint, toppoint = line_wave

    debug.image_lazy(
        lambda: cv2.line(
            cv2ext.convertion_en_couleur(image),
            toppoint,
            bottompoint,
            (0, 0, 255),
            5,
        ),
        DebugImage.Level.TOP,
    )

    angle_ret, posx_ret = compute.get_angle_0_180_posx_safe(
        bottompoint, toppoint
    )
    if posx_ret is None:
        raise Exception("Failed to found vertical line.")
    return angle_ret, posx_ret


def find_best_split_in_all_candidates(
    one: Tuple[Angle, int],
    two: Optional[Tuple[Angle, int]],
    histogram_length: List[Tuple[Angle, int, int]],
    ecart_angle: Angle,
    ecart_posx: int,
) -> Tuple[Angle, int]:
    if two is None:
        return one

    if (
        Angle.abs(one[0] - two[0]) < ecart_angle * 10
        and np.abs(one[1] - two[1]) < 2 * ecart_posx
    ):
        angle_moy = compute.mean_angle([one[0], two[0]])
        pos_moy = (one[1] + two[1]) // 2
        return (angle_moy, pos_moy)

    # Check if angle two is a top in histogram_length.
    possible_line = [
        x
        for x in histogram_length
        if Angle.abs(two[0] - x[0]) < ecart_angle * 10
        and np.abs(two[1] - x[1]) < 2 * ecart_posx
    ]

    # posx of two is the most fiable.
    if len(possible_line) != 0:
        possible_line_transposed = list(zip(*possible_line))
        # Compute mean of possible lines
        angle_moy_one = compute.mean_angle(
            possible_line_transposed[0], possible_line_transposed[2]
        )
        posx_moy_one = compute.mean_weight(
            possible_line_transposed[1], possible_line_transposed[2]
        )

        # and return mean with two
        return (
            compute.mean_angle([angle_moy_one, two[0]]),
            int((posx_moy_one + two[1]) / 2),
        )

    # No candidate ? Maybe their is no line between the two waves.
    return two
