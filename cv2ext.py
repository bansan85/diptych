from typing import Any, List, Tuple, Optional

import cv2
import numpy as np

import compute

if np.__version__.startswith("1.2"):
    # Add typing for numpy :
    # from numpy.typing import ArrayLike.
    # For the moment, they are all Any.
    raise Exception("numpy now support ArrayLike with numpy.typing")


def charge_image(fichier: str) -> Any:
    return cv2.imread(fichier, flags=cv2.IMREAD_ANYDEPTH)


def convertion_en_niveau_de_gris(image: Any) -> Any:
    # Already a 8 bit image.
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def get_polygon_from_contour(contour: Any, number_of_vertices: int) -> Any:
    min_e = 0
    max_e = 1
    max_stagnation = 10

    arc_length_contour = cv2.arcLength(contour, True)

    epsilon = (min_e + max_e) / 2
    epislon_step_dichotomy = (max_e - min_e) / 2
    n_stagnation = 0
    contour_max: List[Any] = []
    last_contour_size = 0

    while True:
        contour_i: Any = cv2.approxPolyDP(
            contour, epsilon * arc_length_contour, True
        )
        if len(contour_i) == number_of_vertices:
            return contour_i
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


def rotate_image(image: Any, angle_deg: float) -> Any:
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle_deg, 1.0)
    result = cv2.warpAffine(
        image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR
    )
    return result


def crop_rectangle(image: Any, crop: Tuple[int, int, int, int]) -> Any:
    return image[crop[2] : crop[3], crop[0] : crop[1]]


def number_channels(image: Any) -> int:
    if image.ndim == 2:
        return 1
    if image.ndim == 3:
        return image.shape[-1]
    raise Exception("Failed to found the number of channels.")


def is_black_white(image: Any) -> bool:
    if number_channels(image) != 1:
        return False
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    return sum(hist[1:255]) < 1


def force_image_to_be_grayscale(
    image: Any, blur_kernel_size: Tuple[int, int], force_blur: bool
) -> Any:
    if number_channels(image) == 1:
        one_channel_image = image.copy()
    else:
        one_channel_image = convertion_en_niveau_de_gris(image)

    if force_blur or is_black_white(one_channel_image):
        return cv2.blur(one_channel_image, blur_kernel_size)
    return one_channel_image


def draw_lines_from_hough_lines(
    image: Any, lines: Any, color: Any, width: int
) -> Any:
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


def get_area(image: Any) -> int:
    return image.shape[0] * image.shape[1]


def get_hw(image: Any) -> Tuple[int, int]:
    return (image.shape[0], image.shape[1])


def remove_border_in_contours(
    contours: Any, border_size: int, image: Any
) -> None:
    height, width = get_hw(image)
    for cnt in contours:
        for contour in cnt:
            if contour[0, 0] - border_size < -1:
                raise ValueError("Contour should not be inside the border.")
            if contour[0, 0] - border_size > width:
                raise ValueError("Contour should not be inside the border.")
            contour[0, 0] = compute.clamp(
                contour[0, 0] - border_size, 0, width - 1
            )
            if contour[0, 1] - border_size < -1:
                raise ValueError("Contour should not be inside the border.")
            if contour[0, 1] - border_size > height:
                raise ValueError("Contour should not be inside the border.")
            contour[0, 1] = compute.clamp(
                contour[0, 1] - border_size, 0, height - 1
            )


def split_image(image: Any, angle: float, posx: int) -> Tuple[Any, Any]:
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
    mask2 = cv2.drawContours(mask, np.int32([pts]), 0, 255, -1)
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


def convertion_en_couleur(image: Any) -> Any:
    # Already a 8 bit image.
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def add_border_to_match_size(
    image: Any,
    paper_size_wh_cm: Tuple[float, float],
    crop: Tuple[int, int, int, int],
    shape_wh: Tuple[int, int],
    dpi: int,
) -> Any:
    height, width = get_hw(image)

    marge_haute_px = crop[2]
    marge_basse_px = shape_wh[1] - 1 - crop[3]
    marge_gauche_px = crop[0]
    marge_droite_px = shape_wh[0] - 1 - crop[1]
    if (
        marge_gauche_px + width + marge_droite_px
        < paper_size_wh_cm[0] / 2.54 * dpi
    ):
        pixels_manquant = paper_size_wh_cm[0] / 2.54 * dpi - width
        left = int(pixels_manquant / 2.0)
        right = int(pixels_manquant / 2.0)
    else:
        raise Exception("marge", "marge_gauche_px")
    if (
        marge_haute_px + height + marge_basse_px
        < paper_size_wh_cm[1] / 2.54 * dpi
    ):
        pixels_manquant = paper_size_wh_cm[1] / 2.54 * dpi - height
        # If no crop at the previous operation, add the same value to the
        # top and the bottom
        if marge_haute_px == 0 and marge_basse_px == 0:
            marge_haute_px = 1
            marge_basse_px = 1
        pourcenthaut = marge_haute_px / (marge_haute_px + marge_basse_px)
        top = int(pixels_manquant * pourcenthaut)
        pourcentbas = marge_basse_px / (marge_haute_px + marge_basse_px)
        bottom = int(pixels_manquant * pourcentbas)
    else:
        raise Exception("marge", "marge_gauche_px")
    return (top, bottom, left, right)


def write_image_if(
    image: Any, enable_debug: Optional[str], filename: str
) -> None:
    if enable_debug is not None:
        cv2.imwrite(enable_debug + filename, image)


def remove_black_border_in_image(
    gray_bordered: Any, enable_debug: Optional[str]
) -> Any:
    gray_bordered2 = cv2.bitwise_not(gray_bordered)
    write_image_if(gray_bordered2, enable_debug, "_2a.png")
    _, threshold = cv2.threshold(gray_bordered, 15, 255, cv2.THRESH_BINARY)
    write_image_if(threshold, enable_debug, "_2b.png")
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    mask_border_only = np.zeros(shape=gray_bordered.shape, dtype=np.uint8)
    height, width = get_hw(gray_bordered)
    for cnt in contours:
        draw = False
        for ptx in cnt:
            point_x, point_y = ptx[0]
            if (
                point_x <= 0
                or point_y <= 0
                or point_x >= width - 1
                or point_y >= height - 1
            ):
                draw = True
                break
        if draw:
            mask_border_only = cv2.drawContours(
                mask_border_only, [cnt], 0, (255), -1
            )

    write_image_if(mask_border_only, enable_debug, "_2c.png")
    res = cv2.bitwise_and(
        gray_bordered2, gray_bordered2, mask=mask_border_only
    )
    write_image_if(res, enable_debug, "_2d.png")
    gray_bordered2 = cv2.bitwise_not(res)
    write_image_if(gray_bordered2, enable_debug, "_2e.png")
    return gray_bordered2


def erode_and_dilate(
    image: Any, size: Tuple[int, int], iterations: int
) -> Any:
    eroded = cv2.erode(
        image,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size),
        iterations=iterations,
    )
    dilate = cv2.dilate(
        eroded,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size),
        iterations=iterations,
    )
    return dilate


def threshold_from_gaussian_histogram(
    image: Any, ksize: int = 5, pourcentage: float = 0.8
) -> Any:
    histogram = compute.image_to_256_histogram(image)
    blur = cv2.GaussianBlur(
        histogram, (1, ksize), ksize, borderType=cv2.BORDER_REPLICATE
    )
    return int(pourcentage * np.argmin(blur))
