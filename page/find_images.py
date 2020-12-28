import types
from typing import List, Optional, Tuple

import cv2
import numpy as np

from angle import Angle
import cv2ext
from debug_image import DebugImage, inc_debug


class FindImageParameters:
    class Impl(types.SimpleNamespace):
        erode_size: int
        kernel_blur_size: Tuple[int, int]
        kernel_morphology_size: Tuple[int, int]
        blur_black_white: Tuple[int, int]
        min_area: float

    def __init__(  # pylint: disable=too-many-arguments
        self,
        erode_size: int,
        kernel_blur_size: Tuple[int, int],
        kernel_morphology_size: Tuple[int, int],
        blur_black_white: Tuple[int, int],
        min_area: float,
    ) -> None:
        self.__param = FindImageParameters.Impl(
            erode_size=erode_size,
            kernel_blur_size=kernel_blur_size,
            kernel_morphology_size=kernel_morphology_size,
            blur_black_white=blur_black_white,
            min_area=min_area,
        )

    @property
    def erode_size(self) -> int:
        return self.__param.erode_size

    @erode_size.setter
    def erode_size(self, val: int) -> None:
        self.__param.erode_size = val

    @property
    def kernel_blur_size(self) -> Tuple[int, int]:
        return self.__param.kernel_blur_size

    @kernel_blur_size.setter
    def kernel_blur_size(self, val: Tuple[int, int]) -> None:
        self.__param.kernel_blur_size = val

    @property
    def kernel_morphology_size(self) -> Tuple[int, int]:
        return self.__param.kernel_morphology_size

    @kernel_morphology_size.setter
    def kernel_morphology_size(self, val: Tuple[int, int]) -> None:
        self.__param.kernel_morphology_size = val

    @property
    def blur_black_white(self) -> Tuple[int, int]:
        return self.__param.blur_black_white

    @blur_black_white.setter
    def blur_black_white(self, val: Tuple[int, int]) -> None:
        self.__param.blur_black_white = val

    @property
    def min_area(self) -> float:
        return self.__param.min_area

    @min_area.setter
    def min_area(self, val: float) -> None:
        self.__param.min_area = val


@inc_debug
def remove_black_border_in_image(
    gray_bordered: np.ndarray, page_angle: Angle, debug: DebugImage
) -> np.ndarray:
    thresholdi = cv2ext.threshold_from_gaussian_histogram_black(gray_bordered)
    _, threshold = cv2.threshold(
        gray_bordered, thresholdi, 255, cv2.THRESH_BINARY_INV
    )
    debug.image(threshold, DebugImage.Level.DEBUG)
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    debug.image_lazy(
        lambda: cv2.drawContours(
            cv2ext.convertion_en_couleur(gray_bordered),
            contours,
            -1,
            (0, 0, 255),
            3,
        ),
        DebugImage.Level.DEBUG,
    )
    __epsilon__ = 5
    mask_border_only = 255 * np.ones(shape=gray_bordered.shape, dtype=np.uint8)
    height, width = cv2ext.get_hw(gray_bordered)
    __angle_tolerance__ = Angle.deg(3.0)
    for cnt in contours:
        (
            (left_top, left_bottom),
            (right_top, right_bottom),
            (top_left, top_right),
            (bottom_left, bottom_right),
        ) = cv2ext.find_longest_lines_in_border(
            (height, width), __epsilon__, cnt
        )

        if (
            left_bottom - left_top > 0
            or top_right - top_left > 0
            or right_bottom - right_top > 0
            or bottom_right - bottom_left > 0
        ):
            cv2ext.insert_border_in_mask(
                cnt,
                threshold,
                mask_border_only,
                (__epsilon__, __angle_tolerance__),
                page_angle,
            )

    # Borders are in black in mask.
    debug.image(mask_border_only, DebugImage.Level.INFO)
    return mask_border_only


@inc_debug
def find_images(
    image: np.ndarray,
    param: FindImageParameters,
    page_angle: Optional[Angle],
    debug: DebugImage,
) -> np.ndarray:
    __internal_border__ = 20
    xxx = 7

    debug.image(image, DebugImage.Level.DEBUG)
    gray = cv2ext.force_image_to_be_grayscale(image, (xxx, xxx))
    debug.image(gray, DebugImage.Level.DEBUG)
    blurimg_bc = cv2ext.erode_and_dilate(gray, (xxx, xxx), xxx)
    debug.image(blurimg_bc, DebugImage.Level.DEBUG)

    if page_angle is not None:
        mask = remove_black_border_in_image(blurimg_bc, page_angle, debug)
        image_no_border = cv2ext.apply_mask(image, mask)
        debug.image(image_no_border, DebugImage.Level.DEBUG)
        gray2 = cv2ext.force_image_to_be_grayscale(image_no_border, (xxx, xxx))
        debug.image(gray2, DebugImage.Level.DEBUG)
        blurimg_bc2 = cv2ext.erode_and_dilate(gray2, (xxx, xxx), xxx, True)
        debug.image(blurimg_bc2, DebugImage.Level.DEBUG)
        gray_no_border = blurimg_bc2
    else:
        gray_no_border = blurimg_bc
    gray_bordered = cv2.copyMakeBorder(
        gray_no_border,
        __internal_border__,
        __internal_border__,
        __internal_border__,
        __internal_border__,
        cv2.BORDER_CONSTANT,
        value=[255],
    )
    debug.image(gray_bordered, DebugImage.Level.DEBUG)
    dilated = cv2.dilate(
        gray_bordered,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (xxx, xxx)),
    )
    debug.image(dilated, DebugImage.Level.DEBUG)
    thresholdi = cv2ext.threshold_from_gaussian_histogram_white(dilated)
    _, threshold = cv2.threshold(
        dilated, thresholdi, 255, cv2.THRESH_BINARY_INV
    )
    debug.image(threshold, DebugImage.Level.DEBUG)

    morpho1 = cv2.morphologyEx(
        threshold,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, param.kernel_morphology_size
        ),
    )
    debug.image(morpho1, DebugImage.Level.DEBUG)
    morpho2 = cv2.morphologyEx(
        morpho1,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, param.kernel_morphology_size
        ),
    )
    debug.image(morpho2, DebugImage.Level.DEBUG)

    contours, _ = cv2.findContours(
        morpho2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = cv2ext.remove_border_in_contours(
        contours, __internal_border__, image
    )
    debug_image_contours = cv2.drawContours(
        cv2ext.convertion_en_couleur(image), contours, -1, (0, 0, 255), 3
    )
    debug.image(debug_image_contours, DebugImage.Level.DEBUG)
    debug_image_mask = np.zeros(cv2ext.get_hw(image), np.uint8)
    img_mask_erode = np.zeros(cv2ext.get_hw(image), np.uint8)
    big_images = filter(
        lambda c: cv2.contourArea(c) > param.min_area * cv2ext.get_area(image),
        contours,
    )
    all_polygon = map(
        lambda cnt: cv2.approxPolyDP(cnt, param.erode_size, True),
        big_images,
    )

    for contour in all_polygon:
        debug_image_contours = cv2.drawContours(
            debug_image_contours, [contour], -1, (255, 0, 0), 3
        )
        debug_image_mask = cv2.drawContours(
            debug_image_mask, [contour], -1, 255, -1
        )
        img_mask_erodei = cv2.drawContours(
            np.zeros(cv2ext.get_hw(image), np.uint8),
            [contour],
            -1,
            (255, 0, 0),
            -1,
        )
        img_mask_erodei = cv2.erode(
            img_mask_erodei,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=param.erode_size,
        )
        img_mask_erode = cv2.bitwise_or(img_mask_erode, img_mask_erodei)

    debug.image(debug_image_contours, DebugImage.Level.DEBUG)
    debug.image(debug_image_mask, DebugImage.Level.DEBUG)
    debug.image(img_mask_erode, DebugImage.Level.INFO)

    return img_mask_erode


def remove_points_inside_images_in_contours(
    contours: List[np.ndarray],
    image: np.ndarray,
    param: FindImageParameters,
    page_angle: Optional[Angle],
    debug: DebugImage,
) -> List[np.ndarray]:
    mask_with_images = find_images(image, param, page_angle, debug)

    contours_filtered = []
    for contour_i in contours:
        # keep point only outside of images found with find_images.
        csi = np.asarray(
            list(
                filter(
                    lambda point_ij: mask_with_images[
                        point_ij[0][1], point_ij[0][0]
                    ]
                    == 0,
                    contour_i,
                )
            )
        )
        if len(csi) == 0:
            continue
        csi.reshape((len(csi), 1, 2))
        contours_filtered.append(csi)
        # Keep only the 3 biggest contours :
        # Two in case of one contour for each page.
        # The third in just for debug.
        if len(contours_filtered) == 3:
            break
    return contours_filtered
