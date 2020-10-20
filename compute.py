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
)
import sys
import time

import numpy as np

_T = TypeVar("_T")

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


def sort_edges_by_posx(line: Tuple[Tuple[int, int], Tuple[int, int]]) -> int:
    _, posx = get_angle_0_180_posx(line[0], line[1])
    if posx is None:
        raise Exception("Line can't be vertical")
    return posx


def get_bottom_point_from_alpha_posx(
    alpha: float, posx: int, height: int
) -> Tuple[int, int]:
    return (
        int(posx - np.tan((alpha - 90.0) / 180.0 * np.pi) * height),
        height - 1,
    )


def get_alpha_posy(
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


def sort_edges_by_posy(line: Tuple[Tuple[int, int], Tuple[int, int]]) -> int:
    _, posy = get_alpha_posy(line[0], line[1])
    if posy is None:
        raise Exception("Line can't be vertical")
    return posy


def get_right_point_from_alpha_posy(
    alpha: float, posy: int, width: int
) -> Tuple[int, int]:
    return width - 1, int(posy + np.tan(alpha / 180.0 * np.pi) * width)


def keep_angle_pos_closed_to_target(
    data: Tuple[float, Optional[int]],
    limit_angle: float,
    target_angle: float,
    target_pos: int,
    limit_pos: int,
) -> bool:
    ang, pos = data
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
        imgw / 200 * 2.54 < width_paper_cm
        and imgh / 200 * 2.54 < height_paper_cm
    ):
        return 200
    if (
        imgw / 300 * 2.54 < width_paper_cm
        and imgh / 300 * 2.54 < height_paper_cm
    ):
        return 300
    raise Exception("dpi", "non détecté")


def find_closed_value(
    histogram: Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]], i: int
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
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


def get_top_histogram(
    smooth: Any, histogram: Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]]
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    retval: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    if smooth[0] > smooth[1]:
        retval.append(find_closed_value(histogram, 0))
    for i in range(1, len(smooth) - 1):
        if smooth[i] > smooth[i - 1] and smooth[i] > smooth[i + 1]:
            retval.append(find_closed_value(histogram, i))
    if smooth[len(smooth) - 1] > smooth[len(smooth) - 2]:
        retval.append(find_closed_value(histogram, len(smooth) - 1))
    return retval
