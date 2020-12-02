import itertools
import math
from typing import (
    Any,
    Iterable,
    Iterator,
    Optional,
    Tuple,
    TypeVar,
    Dict,
    List,
    Union,
)
import sys
import time

import numpy as np
from scipy.stats import norm

_T = TypeVar("_T")
AnyNumber = TypeVar("AnyNumber", int, float)

if np.__version__.startswith("1.2"):
    # Add typing for numpy :
    # from numpy.typing import ArrayLike.
    # For the moment, they are all Any.
    raise Exception("numpy now support ArrayLike with numpy.typing")


def get_angle__180_180(
    point1: Tuple[int, int], point2: Tuple[int, int]
) -> float:
    angle = (
        np.arctan2(point1[1] - point2[1], point1[0] - point2[0]) / np.pi * 180
    )
    return angle


def get_angle_0_180(point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
    angle = get_angle__180_180(point1, point2)
    if angle < 0:
        angle = angle + 180
    return angle


def get_angle_0_180_posx(
    point1: Tuple[int, int], point2: Tuple[int, int]
) -> Tuple[float, Optional[int]]:
    angle = get_angle_0_180(point1, point2)
    if point1[1] == point2[1]:
        posx = None
    else:
        posx = int(
            (point1[0] * point2[1] - point2[0] * point1[1])
            / (point2[1] - point1[1])
        )
    return angle, posx


def get_angle_0_180_posx_safe(
    point1: Tuple[int, int], point2: Tuple[int, int]
) -> Tuple[float, int]:
    angle, posx = get_angle_0_180_posx(point1, point2)
    if posx is None:
        raise Exception("Line can't be horizontal")
    return angle, posx


def get_bottom_point_from_alpha_posx(
    alpha: float, posx: int, height: int
) -> Tuple[int, int]:
    return (
        int(posx - np.tan((alpha - 90.0) / 180.0 * np.pi) * height),
        height - 1,
    )


def get_angle_0_180_posy(
    point1: Tuple[int, int], point2: Tuple[int, int]
) -> Tuple[float, Optional[int]]:
    angle = get_angle_0_180(point1, point2)
    if point1[0] == point2[0]:
        posy = None
    else:
        posy = int(
            (point1[0] * point2[1] - point2[0] * point1[1])
            / (point1[0] - point2[0])
        )
    return angle, posy


def get_angle_0_180_posy_safe(
    point1: Tuple[int, int], point2: Tuple[int, int]
) -> Tuple[float, int]:
    angle, posy = get_angle_0_180_posy(point1, point2)
    if posy is None:
        raise Exception("Line can't be vertical")
    return angle, posy


def get_right_point_from_alpha_posy(
    alpha: float, posy: int, width: int
) -> Tuple[int, int]:
    return width - 1, int(posy + np.tan(alpha / 180.0 * np.pi) * width)


def keep_angle_pos_closed_to_target(
    data: Tuple[int, int, int, int],
    limit_angle: float,
    target_angle: float,
    target_pos: int,
    limit_pos: int,
) -> bool:
    ang, pos = get_angle_0_180_posx((data[0], data[1]), (data[2], data[3]))

    if pos is None:
        return False
    angle_ok = (
        -limit_angle < ang + target_angle and ang + target_angle < limit_angle
    ) or (
        -limit_angle < ang + target_angle - 180
        and ang + target_angle - 180 < limit_angle
    )
    posx_ok = target_pos - limit_pos <= pos <= target_pos + limit_pos
    return angle_ok and posx_ok


def pourcent_error(val1: float, val2: float) -> float:
    if val1 < 0 or val2 < 0:
        raise ValueError("pourcent_error", "rgument must be positive.")
    return np.absolute(val1 - val2) / np.maximum(val1, val2) * 100.0


def iterator_zip_n_n_1(iteration: Iterable[_T]) -> Iterator[Tuple[_T, _T]]:
    iterator = itertools.cycle(iteration)
    next(iterator)

    return zip(iteration, iterator)


def iterator_zip_n_n_2(iteration: Iterable[_T]) -> Iterator[Tuple[_T, _T]]:
    iterator = itertools.cycle(iteration)
    next(iterator)
    next(iterator)

    return zip(iteration, iterator)


def is_contour_rectangle(rectangle: Any, tolerance: float) -> bool:
    if len(rectangle) != 4:
        return False

    distance = [
        math.hypot(i[0, 1] - j[0, 1], i[0, 0] - j[0, 0])
        for i, j in iterator_zip_n_n_1(rectangle)
    ]

    diagonale = [
        math.hypot(i[0, 1] - j[0, 1], i[0, 0] - j[0, 0])
        for i, j in iterator_zip_n_n_2(rectangle)
    ]
    edge1_3 = pourcent_error(distance[0], distance[2]) < tolerance
    edge2_4 = pourcent_error(distance[1], distance[3]) < tolerance
    diag = pourcent_error(diagonale[0], diagonale[1]) < tolerance
    return edge1_3 and edge2_4 and diag


def line_intersection(
    line1: Tuple[Tuple[int, int], Tuple[int, int]],
    line2: Tuple[Tuple[int, int], Tuple[int, int]],
) -> Tuple[int, int]:
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def determinant(point_a: Tuple[int, int], point_b: Tuple[int, int]) -> int:
        return int(
            np.int64(point_a[0]) * point_b[1]
            - np.int64(point_a[1]) * point_b[0]
        )

    div = determinant(xdiff, ydiff)
    if div == 0:
        raise Exception("Lines do not intersect")

    distance = (determinant(*line1), determinant(*line2))
    point_x = determinant(distance, xdiff) / div
    point_y = determinant(distance, ydiff) / div
    return int(point_x), int(point_y)


def clamp(num: Any, min_value: Any, max_value: Any) -> Any:
    return max(min(num, max_value), min_value)


def find_dpi(
    imgw: int, imgh: int, width_paper_cm: float, height_paper_cm: float
) -> int:
    if (
        imgw / 200 * 2.54 < width_paper_cm * 1.1
        and imgh / 200 * 2.54 < height_paper_cm * 1.1
    ):
        return 200
    if (
        imgw / 300 * 2.54 < width_paper_cm * 1.1
        and imgh / 300 * 2.54 < height_paper_cm * 1.1
    ):
        return 300
    raise Exception("dpi", "non détecté")


def find_closed_value(histogram: Dict[int, _T], i: int) -> _T:
    ibis = 0
    while True:
        if i + ibis in histogram:
            return histogram[i + ibis]
        if i - ibis in histogram:
            return histogram[i - ibis]
        ibis = ibis + 1


def optional_concat(root: Optional[str], string: str) -> Optional[str]:
    if root is None:
        return None
    return root + string


def optional_str(condition: bool, string: str) -> Optional[str]:
    if condition:
        return string
    return None


def get_timestamp_ns() -> int:
    if sys.version_info < (3, 7):
        return np.int64(time.time() * 1000000000.0)
    return time.time_ns()  # pylint: disable=no-member,useless-suppression


def get_top_histogram(smooth: Any, histogram: Dict[int, _T]) -> List[_T]:
    retval: List[_T] = []
    if smooth[0] > smooth[1]:
        retval.append(find_closed_value(histogram, 0))
    for i in range(1, len(smooth) - 1):
        if smooth[i] > smooth[i - 1] and smooth[i] > smooth[i + 1]:
            retval.append(find_closed_value(histogram, i))
    if smooth[len(smooth) - 1] > smooth[len(smooth) - 2]:
        retval.append(find_closed_value(histogram, len(smooth) - 1))
    return retval


def get_tops_indices_histogram(smooth: Any) -> List[int]:
    retval: List[int] = []
    if smooth[0] > smooth[1]:
        retval.append(0)
    for i in range(1, len(smooth) - 1):
        # One must be > and the other >= to detect the following top:
        # 50, 51, 51, 51, 50
        if smooth[i] > smooth[i - 1] and smooth[i] >= smooth[i + 1]:
            start = i
            while i < len(smooth) and smooth[i] == smooth[i + 1]:
                i = i + 1
            retval.append((start + i) // 2)
    if smooth[len(smooth) - 1] > smooth[len(smooth) - 2]:
        retval.append(len(smooth) - 1)
    return retval


def norm_cdf(value: float, mean: float, std: float) -> float:
    return norm.cdf(value, mean, std)


def is_angle_closed_to(
    value: float, objectif: float, tolerance: float, max_range: int
) -> bool:
    angle1 = (objectif - tolerance + max_range) % max_range
    angle2 = (objectif + tolerance + max_range) % max_range
    if angle1 <= angle2:
        return angle1 <= value <= angle2
    return value >= angle1 or value <= angle2


def mean_angle(
    liste: Union[List[float], Tuple[float, ...]],
    weight: Optional[Union[List[float], Tuple[float, ...]]] = None,
) -> float:
    if weight is None:
        sin = sum(map(lambda x: np.sin(x / 180.0 * np.pi), liste)) / len(liste)
        cos = sum(map(lambda x: np.cos(x / 180.0 * np.pi), liste)) / len(liste)
    else:
        sin = sum(
            [np.sin(x / 180.0 * np.pi) * y for x, y in zip(liste, weight)]
        ) / sum(
            weight  # type: ignore
        )
        cos = sum(
            [np.cos(x / 180.0 * np.pi) * y for x, y in zip(liste, weight)]
        ) / sum(
            weight  # type: ignore
        )
    if cos < 0:
        return np.arctan(sin / cos) / np.pi * 180.0 + 180.0
    if sin > 0:
        return np.arctan(sin / cos) / np.pi * 180.0
    return np.arctan(sin / cos) / np.pi * 180.0 + 360.0


def mean_weight(
    liste: Union[List[float], Tuple[float, ...]],
    weight: Optional[Union[List[float], Tuple[float, ...]]] = None,
) -> float:
    if weight is None:
        return sum(liste) / len(liste)
    return sum([x * y for x, y in zip(liste, weight)]) / sum(weight)


def get_perpendicular_throught_point(
    line_start: Tuple[AnyNumber, AnyNumber],
    line_end: Tuple[AnyNumber, AnyNumber],
    point: Tuple[AnyNumber, AnyNumber],
) -> Tuple[AnyNumber, AnyNumber]:
    x_1, y_1 = line_start
    x_2, y_2 = line_end
    x_3, y_3 = point

    k = ((y_2 - y_1) * (x_3 - x_1) - (x_2 - x_1) * (y_3 - y_1)) / (
        (y_2 - y_1) ** 2 + (x_2 - x_1) ** 2
    )

    x_4 = x_3 - k * (y_2 - y_1)
    y_4 = y_3 + k * (x_2 - x_1)

    if isinstance(x_1, float):
        return (x_4, y_4)  # type: ignore

    return (int(x_4), int(y_4))


def get_distance_line_point(
    line_start: Tuple[AnyNumber, AnyNumber],
    line_end: Tuple[AnyNumber, AnyNumber],
    point: Tuple[AnyNumber, AnyNumber],
) -> float:
    x_1, y_1 = line_start
    x_2, y_2 = line_end
    x_0, y_0 = point

    return np.abs(
        (y_1 - y_2) * x_0 + (x_2 - x_1) * y_0 + x_1 * y_2 - x_2 * y_1
    ) / np.sqrt((y_2 - y_1) ** 2 + (x_2 - x_1) ** 2)


def hash_djb2_n_3(data: Any) -> int:
    retval = 5381

    for data_1 in data:
        for data_2 in data_1:
            for data_3 in data_2:
                # force int to avoid overflow warning from np.int32
                retval = ((retval << 5) + retval) + int(data_3)

    return retval & 0xFFFFFFFF
