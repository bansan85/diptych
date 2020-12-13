import itertools
import math
from typing import (
    Iterable,
    Iterator,
    Optional,
    Tuple,
    TypeVar,
    Dict,
    List,
    Union,
)
import time

import numpy as np
from scipy.stats import norm

from angle import Angle


_T = TypeVar("_T")
AnyNumber = TypeVar("AnyNumber", int, float)


def get_angle__180_180(
    point1: Tuple[int, int], point2: Tuple[int, int]
) -> Angle:
    return Angle.rad(np.arctan2(point1[1] - point2[1], point1[0] - point2[0]))


def get_angle_0_180(point1: Tuple[int, int], point2: Tuple[int, int]) -> Angle:
    angle = get_angle__180_180(point1, point2)
    if angle.get_rad() < 0:
        angle = angle + Angle.deg(180.0)
    return angle


def get_angle_0_180_posx(
    point1: Tuple[int, int], point2: Tuple[int, int]
) -> Tuple[Angle, Optional[int]]:
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
) -> Tuple[Angle, int]:
    angle, posx = get_angle_0_180_posx(point1, point2)
    if posx is None:
        raise Exception("Line can't be horizontal")
    return angle, posx


def get_bottom_point_from_alpha_posx(
    alpha: Angle, posx: int, height: int
) -> Tuple[int, int]:
    return (
        int(posx - np.tan(alpha.get_rad() - np.pi / 2.0) * height),
        height - 1,
    )


def get_angle_0_180_posy(
    point1: Tuple[int, int], point2: Tuple[int, int]
) -> Tuple[Angle, Optional[int]]:
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
) -> Tuple[Angle, int]:
    angle, posy = get_angle_0_180_posy(point1, point2)
    if posy is None:
        raise Exception("Line can't be vertical")
    return angle, posy


def get_right_point_from_alpha_posy(
    alpha: Angle, posy: int, width: int
) -> Tuple[int, int]:
    return width - 1, int(posy + np.tan(alpha.get_rad()) * width)


def keep_angle_pos_closed_to_target(
    data: Tuple[int, int, int, int],
    limit_angle: Angle,
    target_angle: Angle,
) -> bool:
    ang, pos = get_angle_0_180_posx((data[0], data[1]), (data[2], data[3]))

    if pos is None:
        return False
    angle_ok = (
        -limit_angle < ang + target_angle and ang + target_angle < limit_angle
    ) or (
        -limit_angle < ang + target_angle - Angle.deg(180)
        and ang + target_angle - Angle.deg(180) < limit_angle
    )
    return angle_ok


def pourcent_error(val1: float, val2: float) -> float:
    if val1 < 0 or val2 < 0:
        raise ValueError("pourcent_error", "Argument must be positive.")
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


def is_contour_rectangle(rectangle: np.ndarray, tolerance: float) -> bool:
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
        raise Exception("Lines {0}, {1} do not intersect".format(line1, line2))

    distance = (determinant(*line1), determinant(*line2))
    point_x = determinant(distance, xdiff) / div
    point_y = determinant(distance, ydiff) / div
    return int(point_x), int(point_y)


def convert_line_to_contour(
    line0: Tuple[Tuple[int, int], Tuple[int, int]],
    line1: Tuple[Tuple[int, int], Tuple[int, int]],
    line2: Tuple[Tuple[int, int], Tuple[int, int]],
    line3: Tuple[Tuple[int, int], Tuple[int, int]],
) -> np.ndarray:
    point1_x, point1_y = line_intersection(line0, line2)
    point2_x, point2_y = line_intersection(line0, line3)
    point3_x, point3_y = line_intersection(line1, line2)
    point4_x, point4_y = line_intersection(line1, line3)

    xmoy = (point1_x + point2_x + point3_x + point4_x) // 4
    ymoy = (point1_y + point2_y + point3_y + point4_y) // 4
    list_of_points = [
        [point1_x, point1_y],
        [point2_x, point2_y],
        [point3_x, point3_y],
        [point4_x, point4_y],
    ]

    list_of_points.sort(
        key=lambda x: get_angle__180_180((xmoy, ymoy), (x[0], x[1]))
    )
    return np.asarray(list_of_points)


def clamp(num: int, min_value: int, max_value: int) -> int:
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
    return time.time_ns()


def get_top_histogram(
    smooth: np.ndarray, histogram: Dict[int, _T]
) -> List[_T]:
    retval: List[_T] = []
    if smooth[0] > smooth[1]:
        retval.append(find_closed_value(histogram, 0))
    for i in range(1, len(smooth) - 1):
        if smooth[i] > smooth[i - 1] and smooth[i] > smooth[i + 1]:
            retval.append(find_closed_value(histogram, i))
    if smooth[len(smooth) - 1] > smooth[len(smooth) - 2]:
        retval.append(find_closed_value(histogram, len(smooth) - 1))
    return retval


def get_tops_indices_histogram(smooth: np.ndarray) -> List[int]:
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
    value: Angle, objectif: Angle, tolerance: Angle, max_range: Angle
) -> bool:
    angle1 = (objectif - tolerance + max_range) % max_range
    angle2 = (objectif + tolerance + max_range) % max_range
    if angle1 <= angle2:
        return angle1 <= value <= angle2
    return value >= angle1 or value <= angle2


def atan2(cosinus: float, sinus: float) -> Angle:
    if cosinus < 0:
        return Angle.rad(np.arctan(sinus / cosinus) + np.pi)
    if sinus > 0:
        return Angle.rad(np.arctan(sinus / cosinus))
    return Angle.rad(np.arctan(sinus / cosinus) + 2 * np.pi)


def mean_angle(
    liste: Union[List[Angle], Tuple[Angle, ...]],
    weight: Optional[Union[List[float], Tuple[float, ...]]] = None,
) -> Angle:
    if weight is None:
        sin = sum(map(lambda x: np.sin(x.get_rad()), liste)) / len(liste)
        cos = sum(map(lambda x: np.cos(x.get_rad()), liste)) / len(liste)
    else:
        sin = sum(
            [np.sin(x.get_rad()) * y for x, y in zip(liste, weight)]
        ) / sum(
            weight  # type: ignore
        )
        cos = sum(
            [np.cos(x.get_rad()) * y for x, y in zip(liste, weight)]
        ) / sum(
            weight  # type: ignore
        )
    return atan2(cos, sin)


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


def hash_djb2_n_3(data: np.ndarray) -> int:
    retval = 5381

    for data_1 in data:
        for data_2 in data_1:
            for data_3 in data_2:
                # force int to avoid overflow warning from np.int32
                retval = ((retval << 5) + retval) + int(data_3)

    return retval & 0xFFFFFFFF


def angle_two_lines(
    line1: Tuple[Tuple[int, int], Tuple[int, int]],
    line2: Tuple[Tuple[int, int], Tuple[int, int]],
) -> float:
    angle1 = np.arctan2(line1[0][1] - line1[1][1], line1[0][0] - line1[1][0])
    angle2 = np.arctan2(line2[0][1] - line2[1][1], line2[0][0] - line2[1][0])
    return angle1 - angle2


def line_xy_to_polar(
    line: Tuple[Tuple[int, int], Tuple[int, int]]
) -> Tuple[Angle, float]:
    point = get_perpendicular_throught_point(line[0], line[1], (0, 0))

    angle = get_angle__180_180(point, (0, 0))
    distance = np.sqrt(point[0] ** 2 + point[1] ** 2)

    return (angle, distance)


def angle_between(value: Angle, value_min: Angle, value_max: Angle) -> bool:
    value = value % Angle.deg(360)
    value_min = value_min % Angle.deg(360)
    value_max = value_max % Angle.deg(360)

    if value_min < value_max:
        return value_min <= value <= value_max
    return value_min <= value or value <= value_max


# http://sametmax.com/union-dun-ensemble-dintervalles/
def merge_interval(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    list_debut = [interv[0] for interv in intervals]
    list_fin = [interv[1] for interv in intervals]

    list_debut.sort()
    list_fin.sort()

    list_intervalle_final = []
    nb_superposition = 0
    debut_intervalle_courant = 0

    while list_debut:
        if list_debut[0] < list_fin[0]:
            pos_debut = list_debut.pop(0)
            if nb_superposition == 0:
                debut_intervalle_courant = pos_debut
            nb_superposition += 1
        elif list_debut[0] > list_fin[0]:
            pos_fin = list_fin.pop(0)
            nb_superposition -= 1
            if nb_superposition == 0:
                nouvel_intervalle = (debut_intervalle_courant, pos_fin)
                list_intervalle_final.append(nouvel_intervalle)
        else:
            list_debut.pop(0)
            list_fin.pop(0)

    if list_fin:
        pos_fin = list_fin[-1]
        nouvel_intervalle = (debut_intervalle_courant, pos_fin)
        list_intervalle_final.append(nouvel_intervalle)

    return list_intervalle_final
