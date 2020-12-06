from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np

import compute
from exceptext import NotMyException


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


def remove_error(
    data: np.ndarray, absolute_error: Tuple[float, ...]
) -> np.ndarray:
    if len(data) == 1:
        return data
    _, label, _ = cv2.kmeans(
        data,
        2,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        10,
        cv2.KMEANS_RANDOM_CENTERS,
    )
    ravel = [
        data[label.ravel() == x]
        for x in range(max(label)[0] - min(label)[0] + 1)
    ]
    mean1 = [np.mean(x) for x in list(zip(*ravel[0]))]
    mean2 = [np.mean(x) for x in list(zip(*ravel[1]))]
    effective_error = [
        np.abs(x[0] - x[1]) - x[2]
        for x in list(zip(mean1, mean2, absolute_error))
    ]
    if max(effective_error) < 0:
        return data
    ravel_sorted = sorted(ravel, key=len, reverse=True)

    return ravel_sorted[0]


def get_polygon_from_contour_hough_lines(
    contour: np.ndarray, number_of_vertices: int, image_src: np.ndarray
) -> Optional[np.ndarray]:
    image = np.zeros(get_hw(image_src), dtype=np.uint8)
    image = cv2.drawContours(image, [contour], -1, 255, 1)

    lines_i = cv2.HoughLinesP(
        image,
        1,
        0.05 / 180.0 * np.pi,
        30,
        minLineLength=100,
        maxLineGap=30,
    )

    angle_pos = np.asarray(
        (
            [
                np.asarray(
                    (
                        compute.line_xy_to_polar(
                            ((x[0][0], x[0][1]), (x[0][2], x[0][3]))
                        )
                    ),
                    dtype=np.float32,
                )
                for x in lines_i
            ]
        ),
        dtype=np.float32,
    )

    angle_pos_kmeans = np.asarray(
        (
            [
                np.asarray(
                    (
                        np.cos(((x[0] + 360) % 360) / 180.0 * np.pi * 2),
                        np.sin(((x[0] + 360) % 180) / 180.0 * np.pi * 2),
                        x[1] / min(get_hw(image_src)),
                    ),
                    dtype=np.float32,
                )
                for x in angle_pos
            ]
        ),
        dtype=np.float32,
    )

    _, label, _ = cv2.kmeans(
        angle_pos_kmeans,
        number_of_vertices,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        10,
        cv2.KMEANS_RANDOM_CENTERS,
    )
    ravel = [
        angle_pos[label.ravel() == x]
        for x in range(max(label)[0] - min(label)[0] + 1)
    ]
    ravel_filter = [
        remove_error(x, (0.05 * 180, 0.05 * min(get_hw(image_src))))
        for x in ravel
    ]

    ravel_pre_mean = [
        np.asarray(
            [
                np.asarray(
                    (np.cos(y / 180.0 * np.pi), np.sin(y / 180.0 * np.pi)),
                    dtype=np.float32,
                )
                for y in list(zip(*x))[0]
            ],
            dtype=np.float32,
        )
        for x in ravel_filter
    ]
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
    ravel_pre_mean_4 = [
        compute.angle_between(x[0], x[1] - 180 - 10, x[1] - 180 + 10)
        if len(x) == 2
        else False
        for x in ravel_pre_mean_3
    ]

    ravel_mean = [
        (
            compute.mean_angle([((a + 360) % 360) for a in list(zip(*x))[0]]),
            np.mean(list(zip(*x))[1]),
        )
        if not y
        else (
            compute.mean_angle(
                [
                    a
                    if compute.angle_between(a, z[0] - 10, z[0] + 10)
                    else ((a + 180) % 360)
                    for a in list(zip(*x))[0]
                ]
            ),
            np.mean(
                [
                    b if compute.angle_between(a, z[0] - 10, z[0] + 10) else -b
                    for a, b in x
                ]
            ),
        )
        for x, y, z in list(
            zip(ravel_filter, ravel_pre_mean_4, ravel_pre_mean_3)
        )
    ]
    ravel_points = [
        (
            (
                0 + x[1] * np.cos(x[0] / 180.0 * np.pi),
                0 + x[1] * np.sin(x[0] / 180.0 * np.pi),
            ),
            x[0],
        )
        for x in ravel_mean
    ]
    ravel_lines = [
        (
            x[0],
            (
                x[0][0] + 1000 * np.cos((x[1] + 90) / 180.0 * np.pi),
                x[0][1] + 1000 * np.sin((x[1] + 90) / 180.0 * np.pi),
            ),
            x[1],
        )
        for x in ravel_points
    ]

    lines_sorted = sorted(ravel_lines, key=lambda x: x[2])
    ecart = [
        np.abs(y - x)
        for x, y in compute.iterator_zip_n_n_1(list(zip(*lines_sorted))[2])
    ]
    i = int(np.argmin(ecart))
    if ecart[i] > ecart[(i - 1) % 4] - 5 or ecart[i] > ecart[(i + 1) % 4] - 5:
        return None
    lines_i = compute.convert_line_to_contour(
        (lines_sorted[i % 4][0], lines_sorted[i % 4][1]),
        (lines_sorted[(i + 1) % 4][0], lines_sorted[(i + 1) % 4][1]),
        (lines_sorted[(i + 2) % 4][0], lines_sorted[(i + 2) % 4][1]),
        (lines_sorted[(i + 3) % 4][0], lines_sorted[(i + 3) % 4][1]),
    )
    lines_i2 = np.asarray(
        ([lines_i[0]], [lines_i[1]], [lines_i[2]], [lines_i[3]])
    )

    return lines_i2


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
) -> None:
    height, width = get_hw(image)
    for cnt in contours:
        for contour in cnt:
            contour[0, 0] = compute.clamp(
                contour[0, 0] - border_size, 0, width - 1
            )
            contour[0, 1] = compute.clamp(
                contour[0, 1] - border_size, 0, height - 1
            )


def split_image(
    image: np.ndarray, angle: float, posx: int
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


def secure_write(filename: str, image: np.ndarray) -> None:
    if not cv2.imwrite(filename, image):
        raise NotMyException("Failed to write image " + filename)


def write_image_if(
    image: np.ndarray, enable_debug: Optional[str], filename: str
) -> None:
    if enable_debug is not None:
        secure_write(enable_debug + filename, image)


def __find_longest_lines_in_border(
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


def __insert_border_in_mask(
    cnt: np.ndarray,
    threshold2: np.ndarray,
    mask_border_only: np.ndarray,
    epsilon: Tuple[int, float],
    page_angle: float,
) -> None:
    __pourcentage_white_allowed__ = 0.01
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
                np.linalg.norm(x[0][0] - x[1][0]),
            ),
            all_pair_no_single_pixel,
        )
    )
    vertical_lines = list(
        filter(
            lambda x: compute.is_angle_closed_to(
                x[1], page_angle + 90.0, epsilon_angle, 180
            ),
            all_angles,
        )
    )
    horizontal_lines = list(
        filter(
            lambda x: compute.is_angle_closed_to(
                x[1], page_angle, epsilon_angle, 180
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


def remove_black_border_in_image(
    gray_bordered: np.ndarray, page_angle: float, enable_debug: Optional[str]
) -> np.ndarray:
    thresholdi = threshold_from_gaussian_histogram_black(gray_bordered)
    _, threshold = cv2.threshold(
        gray_bordered, thresholdi, 255, cv2.THRESH_BINARY_INV
    )
    write_image_if(threshold, enable_debug, "_2b2.png")
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    if enable_debug is not None:
        image_contours = cv2.drawContours(
            convertion_en_couleur(gray_bordered), contours, -1, (0, 0, 255), 3
        )
        write_image_if(image_contours, enable_debug, "_2b3.png")
    __epsilon__ = 5
    mask_border_only = 255 * np.ones(shape=gray_bordered.shape, dtype=np.uint8)
    height, width = get_hw(gray_bordered)
    __angle_tolerance__ = 3.0
    for cnt in contours:
        (
            (left_top, left_bottom),
            (right_top, right_bottom),
            (top_left, top_right),
            (bottom_left, bottom_right),
        ) = __find_longest_lines_in_border((height, width), __epsilon__, cnt)

        if (
            left_bottom - left_top > 0
            or top_right - top_left > 0
            or right_bottom - right_top > 0
            or bottom_right - bottom_left > 0
        ):
            __insert_border_in_mask(
                cnt,
                threshold,
                mask_border_only,
                (__epsilon__, __angle_tolerance__),
                page_angle,
            )

    # Borders are in black in mask.
    write_image_if(mask_border_only, enable_debug, "_2c.png")
    return mask_border_only


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
    for i in range(1, 256):
        if histogram_blur[i][0] < histogram_blur[i + 1][0]:
            return i
    return 255


def gaussian_blur_wrap(histogram: np.ndarray, kernel_size: int) -> np.ndarray:
    histogram_wrap = np.concatenate(
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
            np.linalg.norm(
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
            if np.linalg.norm(centers[1] - centers[0]) > __max_distance__:
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
            points = np.concatenate((contours[idx1:], contours[0 : idx2 + 1]))

        new_points = points
        all_checksum: List[int] = []
        checksum = compute.hash_djb2_n_3(new_points)
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

        if len(new_points) == 0:
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
