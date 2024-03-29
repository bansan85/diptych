from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import compute
from .angle import Angle, is_between
from .debug_image import DebugImage
from .parameters import HoughLinesParameters


def charge_image(fichier: str) -> np.ndarray:
    return cv2.imread(fichier, flags=cv2.IMREAD_UNCHANGED)


def convertion_en_niveau_de_gris(image: np.ndarray) -> np.ndarray:
    # Already a 8 bit image.
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def get_polygon_from_contour(
    contour: np.ndarray, number_of_vertices: int
) -> np.ndarray:
    min_e = 0
    max_e = 1
    max_stagnation = 10

    arc_length_contour = cv2.arcLength(contour, True)

    epsilon = (min_e + max_e) / 2
    epislon_step_dichotomy = (max_e - min_e) / 2
    n_stagnation = 0
    contour_max = np.zeros(0)
    last_contour_size = 0

    while True:
        contour_i = cv2.approxPolyDP(
            contour, epsilon * arc_length_contour, True
        )
        if len(contour_i) == number_of_vertices:
            if (
                0.9
                <= cv2.contourArea(contour_i) / cv2.contourArea(contour)
                <= 1 / 0.9
            ):
                return contour_i
            number_of_vertices = number_of_vertices + 1
            epsilon = (min_e + max_e) / 2
            epislon_step_dichotomy = (max_e - min_e) / 2
            n_stagnation = 0
            contour_max = np.zeros(0)
            last_contour_size = 0
            continue
        epislon_step_dichotomy = epislon_step_dichotomy / 2
        if len(contour_i) > number_of_vertices:
            epsilon = epsilon + epislon_step_dichotomy
            n_stagnation = max(n_stagnation + 1, 1)
            if len(contour_max) < number_of_vertices or len(contour_i) < len(
                contour_max
            ):
                contour_max = contour_i
        elif len(contour_i) < number_of_vertices:
            epsilon = epsilon - epislon_step_dichotomy
            n_stagnation = min(n_stagnation - 1, -1)
            # On garde le contour le plus grand au cas où on ne trouve
            # pas un contour de taille suffisant.
            if len(contour_max) < len(contour_i):
                contour_max = contour_i
        if np.abs(n_stagnation) > max_stagnation:
            return contour_max
        # Si la taille du contour change, on réinitialise le compteur.
        if last_contour_size != len(contour_i):
            n_stagnation = 0
        last_contour_size = len(contour_i)


def remove_error(data: List[Any], absolute_error: Tuple[Any, ...]) -> bool:
    if len(data) == 1:
        return False
    data_kmeans = np.asarray(
        [[y / z for y, z in zip(x, absolute_error)] for x in data],
        dtype=np.float32,
    )

    _, label, _ = cv2.kmeans(
        data_kmeans,
        2,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        10,
        cv2.KMEANS_RANDOM_CENTERS,
    )
    ravel = [
        [y for y, z in zip(data_kmeans, label.ravel() == x) if z]
        for x in range(max(label)[0] - min(label)[0] + 1)
    ]
    mean1 = [np.mean(x) for x in list(zip(*ravel[0]))]
    mean2 = [np.mean(x) for x in list(zip(*ravel[1]))]
    effective_error = [np.abs(x[0] - x[1]) for x in list(zip(mean1, mean2))]
    return max(effective_error) >= 1.0


def __found_optimal_kmeans(
    angle_pos: List[Tuple[Angle, float]],
    angle_pos_kmeans: np.ndarray,
    min_shape: int,
) -> List[List[Tuple[Angle, float]]]:
    discretisation = 4
    error = 1
    ravel = []
    while error != 0:
        _, label, _ = cv2.kmeans(
            angle_pos_kmeans,
            discretisation,
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
            10,
            cv2.KMEANS_RANDOM_CENTERS,
        )
        ravel = [
            [y for y, z in zip(angle_pos, label.ravel() == x) if z]
            for x in range(max(label)[0] - min(label)[0] + 1)
        ]
        ravel_filter = [
            remove_error(x, (Angle.deg(0.05 * 180), 0.05 * min_shape))
            for x in ravel
        ]
        error = sum(ravel_filter)
        discretisation += 1

    return ravel


def __found_valid_edge_for_rectangle(
    contour: np.ndarray,
    ravel_lines_mod_90_sorted: List[
        Tuple[Tuple[float, float], Tuple[float, float], Angle, float, Angle]
    ],
    shape_hw: Tuple[int, int],
) -> Optional[Tuple[List[int], List[List[List[int]]]]]:
    current_iter = Angle.deg(10)
    next_iter = Angle.deg(5)
    count_valid_180 = []
    count_valid_180_ok = []
    count_valid_180_sorted = []
    mask_contour = cv2.drawContours(
        np.zeros(shape_hw, dtype=np.uint8), [contour], 0, 255, -1
    )
    area_contour = cv2.contourArea(contour)
    last_ok = None
    nfails = 0
    while nfails != 10:
        count_valid_90 = [
            sum(
                [
                    is_between(
                        y[-1], x[-1], (x[-1] + current_iter) % Angle.deg(90)
                    )
                    for y in ravel_lines_mod_90_sorted
                ]
            )
            for x in ravel_lines_mod_90_sorted
        ]
        count_valid_180 = [
            (
                [
                    y % len(count_valid_90)
                    for y in range(x, x + count_valid_90[x])
                    if is_between(
                        ravel_lines_mod_90_sorted[y % len(count_valid_90)][2]
                        % Angle.deg(180),
                        ravel_lines_mod_90_sorted[x][2] % Angle.deg(180),
                        (ravel_lines_mod_90_sorted[x][2] + current_iter)
                        % Angle.deg(180),
                    )
                ],
                [
                    y % len(count_valid_90)
                    for y in range(x, x + count_valid_90[x])
                    if is_between(
                        ravel_lines_mod_90_sorted[y % len(count_valid_90)][2]
                        % Angle.deg(180),
                        (ravel_lines_mod_90_sorted[x][2] + Angle.deg(90))
                        % Angle.deg(180),
                        (
                            ravel_lines_mod_90_sorted[x][2]
                            + Angle.deg(90)
                            + current_iter
                        )
                        % Angle.deg(180),
                    )
                ],
            )
            for x in range(len(count_valid_90))
        ]

        count_valid_180_sorted = [
            [
                sorted(y, key=lambda z: ravel_lines_mod_90_sorted[z][3])
                for y in x
            ]
            for x in count_valid_180
        ]

        def ternaire(pair: List[List[int]]) -> int:
            if len(pair[0]) < 2 or len(pair[1]) < 2:
                return 0
            rectangle = compute.convert_line_to_contour(
                (
                    ravel_lines_mod_90_sorted[pair[0][0]][0],
                    ravel_lines_mod_90_sorted[pair[0][0]][1],
                ),
                (
                    ravel_lines_mod_90_sorted[pair[0][-1]][0],
                    ravel_lines_mod_90_sorted[pair[0][-1]][1],
                ),
                (
                    ravel_lines_mod_90_sorted[pair[1][0]][0],
                    ravel_lines_mod_90_sorted[pair[1][0]][1],
                ),
                (
                    ravel_lines_mod_90_sorted[pair[1][-1]][0],
                    ravel_lines_mod_90_sorted[pair[1][-1]][1],
                ),
            )
            mask_i = cv2.drawContours(
                255 * np.ones(shape_hw, dtype=np.uint8),
                [rectangle],
                0,
                0,
                -1,
            )
            img_and_i = cv2.bitwise_and(mask_contour, mask_i)
            if (
                # If not enough area, interval too small
                area_contour * 0.99 > cv2.contourArea(rectangle)
                # Too much area outside of the rectangle
                or cv2.countNonZero(img_and_i)
                > 0.1 * mask_i.shape[0] * mask_i.shape[1]
                - cv2.countNonZero(mask_i)
            ):
                return 0
            # Perfect
            if len(pair[0]) == 2 and len(pair[1]) == 2:
                return 1
            # Inverval too big
            if len(pair[0]) >= 2 and len(pair[1]) >= 2:
                return 2
            # Interval too small
            return 0

        count_valid_180_ok = list(map(ternaire, count_valid_180_sorted))
        if sum(count_valid_180_ok) == 0:
            nfails += 1
            current_iter += next_iter
        elif sum(count_valid_180_ok) > 1:
            if next_iter < Angle.deg(0.01):
                break
            current_iter -= next_iter
            nfails = 0
            last_ok = (count_valid_180_ok, count_valid_180_sorted)
        else:
            last_ok = (count_valid_180_ok, count_valid_180_sorted)
            break
        next_iter = Angle.rad(next_iter.get_rad() / 2.0)

    return last_ok


def get_rectangle_from_contour_hough_lines(
    hough_lines_param: HoughLinesParameters,
    contour: np.ndarray,
    image_src: np.ndarray,
    debug: DebugImage,
) -> Optional[np.ndarray]:
    image = np.zeros(get_hw(image_src), dtype=np.uint8)
    image = cv2.drawContours(image, [contour], -1, 255, 1)

    lines_i = cv2.HoughLinesP(
        image,
        hough_lines_param.delta_rho,
        hough_lines_param.delta_tetha.get_rad(),
        hough_lines_param.threshold,
        minLineLength=hough_lines_param.min_line_length,
        maxLineGap=hough_lines_param.max_line_gap,
    )
    debug.image_lazy(
        lambda: draw_lines_from_hough_lines(
            image_src, lines_i, (0, 0, 255), 1
        ),
        DebugImage.Level.DEBUG,
    )

    if len(lines_i) < 4:
        return None

    angle_pos = [
        compute.line_xy_to_polar(((x[0][0], x[0][1]), (x[0][2], x[0][3])))
        for x in lines_i
    ]

    # Sort by inclinaison of the line, not its direction.
    angle_pos_kmeans = np.asarray(
        (
            [
                np.asarray(
                    (
                        np.cos(((x[0].get_rad() + 2.0 * np.pi) % np.pi) * 2.0),
                        np.sin(((x[0].get_rad() + 2.0 * np.pi) % np.pi) * 2.0),
                        x[1] / min(get_hw(image_src)),
                    ),
                    dtype=np.float32,
                )
                for x in angle_pos
            ]
        ),
        dtype=np.float32,
    )

    ravel = __found_optimal_kmeans(
        angle_pos, angle_pos_kmeans, min(get_hw(image_src))
    )

    ravel_pre_mean = [
        np.asarray(
            [
                np.asarray(
                    (np.cos(y.get_rad()), np.sin(y.get_rad())),
                    dtype=np.float32,
                )
                for y in list(zip(*x))[0]
            ],
            dtype=np.float32,
        )
        for x in ravel
    ]
    # For each inclinaison of line, check if direction is different.
    ravel_pre_mean_2 = [
        cv2.kmeans(
            x,
            2,
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
            10,
            cv2.KMEANS_RANDOM_CENTERS,
        )
        if len(x) >= 2
        else (0.0, np.asarray([[0]], dtype=np.int32), x)
        for x in ravel_pre_mean
    ]
    ravel_pre_mean_label = [
        [x[y[1].ravel() == z] for z in range(max(y[1])[0] - min(y[1])[0] + 1)]
        for x, y in zip(ravel_pre_mean, ravel_pre_mean_2)
    ]
    ravel_pre_mean_3 = [
        [compute.mean_angle([compute.atan2(x[0], x[1]) for x in z]) for z in y]
        for y in ravel_pre_mean_label
    ]
    # True if there is line with same inclinaison but both direction
    ravel_pre_mean_4 = [
        is_between(x[0], x[1] - Angle.deg(190), x[1] - Angle.deg(170))
        if len(x) == 2
        else False
        for x in ravel_pre_mean_3
    ]

    ravel_mean = [
        (
            compute.mean_angle(
                [
                    ((a + Angle.deg(360)) % Angle.deg(360))
                    for a in list(zip(*x))[0]
                ]
            ),
            float(np.mean(list(zip(*x))[1])),
        )
        if not y
        else (
            compute.mean_angle(
                [
                    a
                    if is_between(
                        a, z[0] - Angle.deg(10), z[0] + Angle.deg(10)
                    )
                    else ((a + Angle.deg(180)) % Angle.deg(360))
                    for a in list(zip(*x))[0]
                ]
            ),
            float(
                np.mean(
                    [
                        b
                        if is_between(
                            a, z[0] - Angle.deg(10), z[0] + Angle.deg(10)
                        )
                        else -b
                        for a, b in x
                    ]
                )
            ),
        )
        for x, y, z in list(zip(ravel, ravel_pre_mean_4, ravel_pre_mean_3))
    ]
    ravel_points = [
        (
            (
                0 + x[1] * np.cos(x[0].get_rad()),
                0 + x[1] * np.sin(x[0].get_rad()),
            ),
            x[0],
            x[1],
        )
        for x in ravel_mean
    ]
    ravel_lines = [
        (
            x[0],
            (
                x[0][0] + 1000 * np.cos(x[1].get_rad() + np.pi / 2),
                x[0][1] + 1000 * np.sin(x[1].get_rad() + np.pi / 2),
            ),
            x[1],
            x[2],
        )
        for x in ravel_points
    ]

    ravel_lines_mod_90 = [(*x, x[2] % Angle.deg(90)) for x in ravel_lines]
    ravel_lines_mod_90_sorted = sorted(ravel_lines_mod_90, key=lambda x: x[-1])

    retval = __found_valid_edge_for_rectangle(
        contour, ravel_lines_mod_90_sorted, get_hw(image_src)
    )

    if retval is None:
        return None

    count_valid_180_ok, count_valid_180_sorted = retval

    max_area = 0
    final_lines = np.empty((0))
    lines_i2 = np.empty((0))
    for _, lines in filter(
        lambda x: x[0] >= 1, zip(count_valid_180_ok, count_valid_180_sorted)
    ):
        lines_i2 = compute.convert_line_to_contour(
            (
                ravel_lines_mod_90_sorted[lines[0][0]][0],
                ravel_lines_mod_90_sorted[lines[0][0]][1],
            ),
            (
                ravel_lines_mod_90_sorted[lines[0][-1]][0],
                ravel_lines_mod_90_sorted[lines[0][-1]][1],
            ),
            (
                ravel_lines_mod_90_sorted[lines[1][0]][0],
                ravel_lines_mod_90_sorted[lines[1][0]][1],
            ),
            (
                ravel_lines_mod_90_sorted[lines[1][-1]][0],
                ravel_lines_mod_90_sorted[lines[1][-1]][1],
            ),
        )
        lines_i3 = np.asarray(
            ([lines_i2[0]], [lines_i2[1]], [lines_i2[2]], [lines_i2[3]])
        )
        area_i = cv2.contourArea(lines_i3)
        if max_area == 0 or area_i < max_area:
            max_area = area_i
            final_lines = lines_i3

    return final_lines


def rotate_image(image: np.ndarray, angle_deg: float) -> np.ndarray:
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle_deg, 1.0)
    result = cv2.warpAffine(
        image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR
    )
    return result


def crop_rectangle(
    image: np.ndarray, crop: Tuple[int, int, int, int]
) -> np.ndarray:
    return image[
        compute.clamp(crop[2], 0, len(image) - 1) : compute.clamp(
            crop[3], 0, len(image) - 1
        ),
        compute.clamp(crop[0], 0, len(image[0]) - 1) : compute.clamp(
            crop[1], 0, len(image[0]) - 1
        ),
    ]


def number_channels(image: np.ndarray) -> int:
    if image.ndim == 2:
        return 1
    if image.ndim == 3:
        return image.shape[-1]
    raise Exception("Failed to found the number of channels.")


def force_image_to_be_grayscale(
    image: np.ndarray, blur_kernel_size: Tuple[int, int]
) -> np.ndarray:
    if number_channels(image) == 1:
        one_channel_image = image.copy()
    else:
        one_channel_image = convertion_en_niveau_de_gris(image)

    return cv2.blur(one_channel_image, blur_kernel_size)


def draw_lines_from_hough_lines(
    image: np.ndarray,
    lines: np.ndarray,
    color: Tuple[int, int, int],
    width: int,
) -> np.ndarray:
    image_with_lines = convertion_en_couleur(image)
    for line in lines:
        for point1_x, point1_y, point2_x, point2_y in line:
            cv2.line(
                image_with_lines,
                (point1_x, point1_y),
                (point2_x, point2_y),
                color,
                width,
            )
    return image_with_lines


def get_area(image: np.ndarray) -> int:
    return image.shape[0] * image.shape[1]


def get_hw(image: np.ndarray) -> Tuple[int, int]:
    return (image.shape[0], image.shape[1])


def remove_border_in_contours(
    contours: List[np.ndarray], border_size: int, image: np.ndarray
) -> List[np.ndarray]:
    height, width = get_hw(image)
    height -= 1
    width -= 1

    def subst(contour: np.ndarray) -> np.ndarray:
        contour = contour - border_size
        contour[:, 0, 0] = np.clip(contour[:, 0, 0], 0, width)
        contour[:, 0, 1] = np.clip(contour[:, 0, 1], 0, height)
        return contour

    return list(map(subst, contours))


def split_image(
    image: np.ndarray, angle: Angle, posx: int
) -> Tuple[np.ndarray, np.ndarray]:
    height, width = get_hw(image)

    toppoint = (posx, 0)
    bottompoint = compute.get_bottom_point_from_alpha_posx(angle, posx, height)

    # On défini le masque pour séparer la page droite et gauche
    mask = np.zeros((height, width), np.uint8)
    pts = np.array(
        [
            [0, 0],
            [toppoint[0], 0],
            [toppoint[0], toppoint[1]],
            [bottompoint[0], bottompoint[1]],
            [bottompoint[0], height - 1],
            [0, height - 1],
        ]
    )
    mask2 = cv2.drawContours(
        mask, np.asarray([pts], dtype=np.int32), 0, 255, -1
    )
    page_gauche = image.copy()
    page_droite = image.copy()
    # On applique le masque
    page_gauche[mask2 == 0] = 0
    page_droite[mask2 > 0] = 0
    # On crop les images.
    page_gauche_0 = crop_rectangle(
        page_gauche,
        (0, np.maximum(toppoint[0], bottompoint[0]) - 1, 0, height - 1),
    )
    page_droite_0 = crop_rectangle(
        page_droite,
        (np.minimum(toppoint[0], bottompoint[0]), width, 0, height - 1),
    )

    # On renvoie les images cropées.
    return page_gauche_0, page_droite_0


def convertion_en_couleur(image: np.ndarray) -> np.ndarray:
    # Already a 8 bit image.
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def add_border_to_match_size(
    image: np.ndarray,
    paper_size_wh_cm: Tuple[float, float],
    crop: Tuple[int, int, int, int],
    shape_wh: Tuple[int, int],
    dpi: int,
) -> Tuple[int, int, int, int]:
    height, width = get_hw(image)

    marge_haute_px = crop[2]
    marge_basse_px = shape_wh[1] - 1 - crop[3]

    pixels_manquant = paper_size_wh_cm[0] / 2.54 * dpi - width
    if pixels_manquant < 0:
        raise Exception("marge", "marge_gauche_px")
    left = int(pixels_manquant / 2.0)
    right = int(pixels_manquant / 2.0)

    pixels_manquant = paper_size_wh_cm[1] / 2.54 * dpi - height
    if pixels_manquant < 0:
        raise Exception("marge", "marge_haute_px")

    # If no crop at the previous operation, add the same value to the
    # top and the bottom
    if marge_haute_px == 0 and marge_basse_px == 0:
        marge_haute_px = 1
        marge_basse_px = 1
    pourcenthaut = marge_haute_px / (marge_haute_px + marge_basse_px)
    top = int(pixels_manquant * pourcenthaut)
    pourcentbas = marge_basse_px / (marge_haute_px + marge_basse_px)
    bottom = int(pixels_manquant * pourcentbas)

    return (top, bottom, left, right)


def find_longest_lines_in_border(
    shape: Tuple[int, int], epsilon: int, cnt: np.ndarray
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    height, width = shape
    left_top = height
    left_bottom = 0
    right_top = height
    right_bottom = 0
    top_left = width
    top_right = 0
    bottom_left = width
    bottom_right = 0
    for pt1, pt2 in compute.iterator_zip_n_n_1(cnt):
        point1_x, point1_y = pt1[0]
        point2_x, point2_y = pt2[0]
        if point1_x <= epsilon and point2_x <= epsilon:
            left_top = min(left_top, point1_y, point2_y)
            left_bottom = max(left_bottom, point1_y, point2_y)
        if point1_y <= epsilon and point2_y <= epsilon:
            top_left = min(top_left, point1_x, point2_x)
            top_right = max(top_right, point1_x, point2_x)
        if point1_x >= width - 1 - epsilon and point2_x >= width - 1 - epsilon:
            right_top = min(right_top, point1_y, point2_y)
            right_bottom = max(right_bottom, point1_y, point2_y)
        if (
            point1_y >= height - 1 - epsilon
            and point2_y >= height - 1 - epsilon
        ):
            bottom_left = min(bottom_left, point1_x, point2_x)
            bottom_right = max(bottom_right, point1_x, point2_x)
    return (
        (left_top, left_bottom),
        (right_top, right_bottom),
        (top_left, top_right),
        (bottom_left, bottom_right),
    )


def insert_border_in_mask(
    cnt: np.ndarray,
    threshold2: np.ndarray,
    mask_border_only: np.ndarray,
    epsilon: Tuple[int, Angle],
    page_angle: Angle,
) -> None:
    __pourcentage_white_allowed__ = 0.015
    epsilon_border, epsilon_angle = epsilon
    height, width = get_hw(threshold2)
    cnt2 = cnt[cnt[:, 0, 0] > epsilon_border]
    cnt3 = cnt2[cnt2[:, 0, 0] < width - 1 - epsilon_border]
    cnt4 = cnt3[cnt3[:, 0, 1] > epsilon_border]
    cnt5 = cnt4[cnt4[:, 0, 1] < height - 1 - epsilon_border]
    if len(cnt5) == 0:
        return
    contour_approximate = cv2.approxPolyDP(cnt5, epsilon_border, True)
    all_pair = list(compute.iterator_zip_n_n_1(contour_approximate))
    all_pair_no_single_pixel = list(
        filter(
            lambda x: x[0][0][0] != x[1][0][0] or x[0][0][1] != x[1][0][1],
            all_pair,
        )
    )
    all_angles = list(
        map(
            lambda x: (
                (x[0][0], x[1][0]),
                compute.get_angle_0_180(x[0][0], x[1][0]),
                np.linalg.norm(x[0][0] - x[1][0]),  # type: ignore
            ),
            all_pair_no_single_pixel,
        )
    )
    vertical_lines = list(
        filter(
            lambda x: compute.is_angle_closed_to(
                x[1],
                page_angle + Angle.deg(90.0),
                epsilon_angle,
                Angle.deg(180),
            ),
            all_angles,
        )
    )
    horizontal_lines = list(
        filter(
            lambda x: compute.is_angle_closed_to(
                x[1], page_angle, epsilon_angle, Angle.deg(180)
            ),
            all_angles,
        )
    )
    vertical_lines_pos = list(
        map(
            lambda x: (
                compute.get_angle_0_180_posx_safe(x[0][0], x[0][1])[1],
                x[1],
            ),
            vertical_lines,
        )
    )
    horizontal_lines_pos = list(
        map(
            lambda x: (
                compute.get_angle_0_180_posy_safe(x[0][0], x[0][1])[1],
                x[1],
            ),
            horizontal_lines,
        )
    )
    vertical_lines_pos.sort(key=lambda x: x[0])
    horizontal_lines_pos.sort(key=lambda x: x[0])
    for posx, angle in vertical_lines_pos:
        mask = np.zeros((height, width), np.uint8)
        bottom_point = compute.get_bottom_point_from_alpha_posx(
            angle, posx, height
        )
        if posx < width / 2:
            pts = np.array(
                [
                    [-1, 0],
                    [posx - 1, 0],
                    [bottom_point[0] - 1, bottom_point[1]],
                    [-1, height - 1],
                ]
            )
        else:
            pts = np.array(
                [
                    [width, 0],
                    [posx + 1, 0],
                    [bottom_point[0] + 1, bottom_point[1]],
                    [width, height - 1],
                ]
            )
        mask = cv2.drawContours(mask, [pts], 0, 255, -1)
        histogram = cv2.calcHist([threshold2], [0], mask, [2], [0, 256])
        if __pourcentage_white_allowed__ * histogram[0] > sum(
            histogram[1:]
        ) or __pourcentage_white_allowed__ * histogram[-1] > sum(
            histogram[:-1]
        ):
            mask_border_only = cv2.drawContours(
                mask_border_only, [pts], 0, (0), -1
            )
    for posy, angle in horizontal_lines_pos:
        mask = np.zeros((height, width), np.uint8)
        bottom_point = compute.get_right_point_from_alpha_posy(
            angle, posy, width
        )
        if posy < height / 2:
            pts = np.array(
                [
                    [0, -1],
                    [0, posy - 1],
                    [bottom_point[0], bottom_point[1] - 1],
                    [width - 1, -1],
                ]
            )
        else:
            pts = np.array(
                [
                    [0, height],
                    [0, posy + 1],
                    [bottom_point[0], bottom_point[1] + 1],
                    [width - 1, height],
                ]
            )
        mask = cv2.drawContours(mask, [pts], 0, 255, -1)
        histogram = cv2.calcHist([threshold2], [0], mask, [2], [0, 256])
        if __pourcentage_white_allowed__ * histogram[0] > sum(
            histogram[1:]
        ) or __pourcentage_white_allowed__ * histogram[-1] > sum(
            histogram[:-1]
        ):
            mask_border_only = cv2.drawContours(
                mask_border_only, [pts], 0, (0), -1
            )


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    gray_bordered2 = cv2.bitwise_not(image)
    gray_bordered3 = cv2.bitwise_and(gray_bordered2, gray_bordered2, mask=mask)
    gray_bordered4 = cv2.bitwise_not(gray_bordered3)
    # Borders are in white in original image.
    return gray_bordered4


def erode_and_dilate(
    image: np.ndarray,
    size: Tuple[int, int],
    iterations: int,
    reverse: bool = False,
) -> np.ndarray:
    start = int(reverse)
    img = image
    for i in range(2):
        if (i + start) % 2 == 0:
            img = cv2.erode(
                img,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size),
                iterations=iterations,
            )
        else:
            img = cv2.dilate(
                img,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size),
                iterations=iterations,
            )
    return img


def threshold_from_gaussian_histogram_white(
    image: np.ndarray, pourcentage: float = 0.2, blur_kernel_size: int = 31
) -> int:
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    histogram_blur = cv2.GaussianBlur(
        histogram,
        (1, blur_kernel_size),
        blur_kernel_size,
        borderType=cv2.BORDER_REPLICATE,
    )
    i = 255
    extreme_min = histogram_blur[255][0]
    for j in range(254, 0, -1):
        if histogram_blur[j][0] < extreme_min:
            extreme_min = histogram_blur[j][0]
        else:
            i = j
            break
    limit = extreme_min * (1 + pourcentage)
    for j in range(i, 0, -1):
        if histogram_blur[j][0] > limit:
            i = j
            break
    return i


def threshold_from_gaussian_histogram_black(
    image: np.ndarray, blur_kernel_size: int = 31
) -> int:
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    histogram_blur = cv2.GaussianBlur(
        histogram,
        (1, blur_kernel_size),
        blur_kernel_size,
        borderType=cv2.BORDER_REPLICATE,
    )
    for i in range(0, 255):
        if histogram_blur[i][0] < histogram_blur[i + 1][0]:
            return i
    return 255


def gaussian_blur_wrap(histogram: np.ndarray, kernel_size: int) -> np.ndarray:
    histogram_wrap = np.concatenate(  # type: ignore
        [
            histogram[-kernel_size:],
            histogram,
            histogram[:kernel_size],
        ]
    )
    histogram_wrap_blur = cv2.GaussianBlur(
        histogram_wrap,
        (1, kernel_size),
        kernel_size,
        borderType=cv2.BORDER_REPLICATE,
    )
    return histogram_wrap_blur[kernel_size:-kernel_size]


def apply_brightness_contrast(
    input_img: np.ndarray, brightness: int = 0, contrast: int = 0
) -> np.ndarray:
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        alpha_c = 131 * (contrast + 127) / (127 * (131 - contrast))
        gamma_c = 127 * (1 - alpha_c)
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


# return cv2ext.bounding_rectangle(
#     cv2ext.get_hw(images_mask),
#     (lines_vertical_angle, lines_horizontal_angle),
#     (flag_v_min, flag_v_max, flag_h_min, flag_h_max),
# )
def bounding_rectangle(
    shape: Tuple[int, int],
    lines: Tuple[
        List[Tuple[Tuple[int, int], Tuple[int, int]]],
        List[Tuple[Tuple[int, int], Tuple[int, int]]],
    ],
    flags: Tuple[List[bool], List[bool], List[bool], List[bool]],
) -> np.ndarray:
    mask = 255 * np.ones(shape, dtype=np.uint8)
    lines_vertical_angle, lines_horizontal_angle = lines

    for line, flag in zip(lines_vertical_angle, flags[0]):
        if not flag:
            continue
        pt1, pt2 = line
        angle, posx = compute.get_angle_0_180_posx_safe(pt1, pt2)

        pts = np.array(
            [
                [0, 0],
                [posx, 0],
                compute.get_bottom_point_from_alpha_posx(
                    angle, posx, shape[0]
                ),
                [0, shape[0] - 1],
            ]
        )
        mask = cv2.drawContours(
            mask, np.asarray([pts], dtype=np.int32), 0, (0), -1
        )

    for line, flag in zip(lines_vertical_angle, flags[1]):
        if not flag:
            continue
        pt1, pt2 = line
        angle, posx = compute.get_angle_0_180_posx_safe(pt1, pt2)

        pts = np.array(
            [
                [shape[1] - 1, 0],
                [posx, 0],
                compute.get_bottom_point_from_alpha_posx(
                    angle, posx, shape[0]
                ),
                [shape[1] - 1, shape[0] - 1],
            ]
        )
        mask = cv2.drawContours(
            mask, np.asarray([pts], dtype=np.int32), 0, (0), -1
        )

    for line, flag in zip(lines_horizontal_angle, flags[2]):
        if not flag:
            continue
        pt1, pt2 = line
        angle, posy = compute.get_angle_0_180_posy_safe(pt1, pt2)

        pts = np.array(
            [
                [0, 0],
                [0, posy],
                compute.get_right_point_from_alpha_posy(angle, posy, shape[1]),
                [shape[1] - 1, 0],
            ]
        )
        mask = cv2.drawContours(
            mask, np.asarray([pts], dtype=np.int32), 0, (0), -1
        )

    for line, flag in zip(lines_horizontal_angle, flags[3]):
        if not flag:
            continue
        pt1, pt2 = line
        angle, posy = compute.get_angle_0_180_posy_safe(pt1, pt2)

        pts = np.array(
            [
                [0, shape[0] - 1],
                [0, posy],
                compute.get_right_point_from_alpha_posy(angle, posy, shape[1]),
                [shape[1] - 1, shape[0] - 1],
            ]
        )
        mask = cv2.drawContours(
            mask, np.asarray([pts], dtype=np.int32), 0, (0), -1
        )

    rectangle = cv2.boundingRect(mask)

    return np.array(
        [
            [[rectangle[0], rectangle[1]]],
            [[rectangle[0], rectangle[1] + rectangle[3]]],
            [[rectangle[0] + rectangle[2], rectangle[1] + rectangle[3]]],
            [[rectangle[0] + rectangle[2], rectangle[1]]],
        ]
    )


def remove_perpendicular_multiples_points(
    line_res: Tuple[float, float, float, float], points: np.ndarray
) -> np.ndarray:
    v_x, v_y, c_x, c_y = line_res

    points_dict: Dict[int, List[np.ndarray]] = {}
    for point_i in points:
        projection = compute.get_perpendicular_throught_point(
            (c_x, c_y), (c_x + v_x, c_y + v_y), point_i[0]
        )
        length = int(
            np.linalg.norm(  # type: ignore
                np.array((c_x, c_y)) - np.array((projection[0], projection[1]))
            )
        )
        angle = np.arctan2(projection[0] - c_x, projection[1] - c_y)
        if angle == 0.0:
            angle = -1
        angle_sign = int(np.sign(angle))
        points_dict[length * angle_sign] = points_dict.get(
            length * angle_sign, []
        )
        points_dict[length * angle_sign].append(point_i)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        10,
        1.0,
    )
    flags = cv2.KMEANS_RANDOM_CENTERS
    __max_distance__ = 5
    for key in list(points_dict):
        if len(points_dict[key]) > 1:
            _, _, centers = cv2.kmeans(
                np.array(points_dict[key], dtype=np.float32),
                2,
                None,
                criteria,
                10,
                flags,
            )
            if (
                np.linalg.norm(centers[1] - centers[0])  # type: ignore
                > __max_distance__
            ):
                del points_dict[key]

    list_new_points = []
    for key in list(points_dict):
        list_new_points.append(points_dict[key][0])

    return np.asarray(list_new_points)


def convert_polygon_with_fitline(
    contours: np.ndarray, polygon: np.ndarray
) -> Tuple[List[Tuple[Tuple[int, int], Tuple[int, int]]], List[np.ndarray]]:
    ret = []
    contours_list = contours.tolist()
    index_of_poly = [
        contours_list.index([[polygon_i[0][0], polygon_i[0][1]]])
        for polygon_i in polygon
    ]
    ret_points = []

    __max_iterations__ = 5
    __max_ecart__ = 50
    __min_valid_ecart__ = 70

    for idx1, idx2 in compute.iterator_zip_n_n_1(index_of_poly):
        if idx1 < idx2:
            points = contours[idx1 : idx2 + 1]
        else:
            points = np.concatenate(
                (contours[idx1:], contours[0 : idx2 + 1])
            )  # type: ignore

        new_points = points
        all_checksum: List[int] = []
        checksum = compute.hash_djb2_n_3(new_points)
        line_res = None
        while checksum not in all_checksum:
            all_checksum.append(checksum)
            line_res = cv2.fitLine(new_points, cv2.DIST_L2, 0, 0.01, 0.01)
            new_points = remove_perpendicular_multiples_points(
                (
                    line_res[0][0],
                    line_res[1][0],
                    line_res[2][0],
                    line_res[3][0],
                ),
                points,
            )
            if len(all_checksum) == __max_iterations__:
                break

            if len(new_points) == 0:
                break
            checksum = compute.hash_djb2_n_3(new_points)

        if len(new_points) == 0 or line_res is None:
            continue

        all_perpendicular_distances = [
            compute.get_distance_line_point(
                (line_res[2][0], line_res[3][0]),
                (
                    line_res[2][0] + line_res[0][0],
                    line_res[3][0] + line_res[1][0],
                ),
                (x[0][0], x[0][1]),
            )
            for x in new_points
        ]
        valid_distances = list(
            filter(lambda x: x < __max_ecart__, all_perpendicular_distances)
        )

        if (
            len(valid_distances) * 100 / len(all_perpendicular_distances)
            < __min_valid_ecart__
        ):
            return ([], [])

        v_x, v_y, c_x, c_y = line_res

        if abs(v_x) > abs(v_y):
            min_x = min(points[:, 0, 0])
            max_x = max(points[:, 0, 0])
            lefty = int(((-c_x + min_x) * v_y / v_x) + c_y)
            righty = int(((max_x - c_x) * v_y / v_x) + c_y)
            line = ((min_x, lefty), (max_x, righty))
        else:
            min_y = min(points[:, 0, 1])
            max_y = max(points[:, 0, 1])
            topx = int(((-c_y + min_y) * v_x / v_y) + c_x)
            bottomx = int(((max_y - c_y) * v_x / v_y) + c_x)
            line = ((topx, min_y), (bottomx, max_y))

        for point in new_points:
            ret_points.append(point[0])
        ret.append(line)

    return (ret, ret_points)


def best_fitline(
    point_1a: Tuple[int, int],
    point_1b: Tuple[int, int],
    valid_lines: List[Tuple[int, int, int, int]],
    shape_hw: Tuple[int, int],
    distance: int,
) -> Tuple[Angle, int]:
    image = np.zeros((shape_hw), dtype=np.uint8)

    for line in valid_lines:
        cv2.line(
            image,
            (line[0], line[1]),
            (line[2], line[3]),
            255,
            1,
        )

    indices = np.where(image == 255)

    points = list(
        filter(
            lambda point: compute.get_distance_line_point(
                point_1a,
                point_1b,
                (point[1], point[0]),
            )
            < distance,
            zip(*indices),
        )
    )

    line_res = cv2.fitLine(np.asarray(points), cv2.DIST_L2, 0, 0.01, 0.01)

    return compute.get_angle_0_180_posx_safe(
        (line_res[3][0], line_res[2][0]),
        (line_res[3][0] + line_res[1][0], line_res[2][0] + line_res[0][0]),
    )
