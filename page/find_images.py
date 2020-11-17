import types
from typing import Tuple, Optional, Any, List

import numpy as np

import cv2
import cv2ext


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
    ):
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


def find_images(
    image: Any,
    param: FindImageParameters,
    page_angle: Optional[float],
    enable_debug: Optional[str],
) -> Any:
    __internal_border__ = 20
    xxx = 7

    cv2ext.write_image_if(image, enable_debug, "_1a.png")
    gray = cv2ext.force_image_to_be_grayscale(image, (xxx, xxx), True)
    cv2ext.write_image_if(gray, enable_debug, "_1b.png")
    blurimg_bc = cv2ext.erode_and_dilate(gray, (xxx, xxx), xxx)
    cv2ext.write_image_if(blurimg_bc, enable_debug, "_1c.png")

    if page_angle is not None:
        mask = cv2ext.remove_black_border_in_image(
            blurimg_bc, page_angle, enable_debug
        )
        image_no_border = cv2ext.apply_mask(image, mask)
        cv2ext.write_image_if(image_no_border, enable_debug, "_2d.png")
        gray2 = cv2ext.force_image_to_be_grayscale(
            image_no_border, (xxx, xxx), True
        )
        cv2ext.write_image_if(gray2, enable_debug, "_2e.png")
        blurimg_bc2 = cv2ext.erode_and_dilate(gray2, (xxx, xxx), xxx, True)
        cv2ext.write_image_if(blurimg_bc2, enable_debug, "_2f.png")
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
    cv2ext.write_image_if(gray_bordered, enable_debug, "_3b.png")
    dilated = cv2.dilate(
        gray_bordered,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (xxx, xxx)),
    )
    cv2ext.write_image_if(dilated, enable_debug, "_3c.png")
    thresholdi = cv2ext.threshold_from_gaussian_histogram_white(dilated)
    _, threshold = cv2.threshold(
        dilated, thresholdi, 255, cv2.THRESH_BINARY_INV
    )
    cv2ext.write_image_if(threshold, enable_debug, "_4.png")

    morpho1 = cv2.morphologyEx(
        threshold,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, param.kernel_morphology_size
        ),
    )
    cv2ext.write_image_if(morpho1, enable_debug, "_6.png")
    morpho2 = cv2.morphologyEx(
        morpho1,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, param.kernel_morphology_size
        ),
    )
    cv2ext.write_image_if(morpho2, enable_debug, "_7.png")

    contours, _ = cv2.findContours(
        morpho2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2ext.remove_border_in_contours(contours, __internal_border__, image)
    if enable_debug is not None:
        debug_image_contours = cv2.drawContours(
            cv2ext.convertion_en_couleur(image), contours, -1, (0, 0, 255), 3
        )
        cv2ext.secure_write(enable_debug + "_8.png", debug_image_contours)
        debug_image_mask = np.zeros(image.shape, np.uint8)
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
        if enable_debug is not None:
            debug_image_contours = cv2.drawContours(
                debug_image_contours, [contour], -1, (255, 0, 0), 3
            )
            debug_image_mask = cv2.drawContours(
                debug_image_mask, [contour], -1, (255, 0, 0), -1
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

    if enable_debug is not None:
        cv2ext.secure_write(enable_debug + "_9.png", debug_image_contours)
        cv2ext.secure_write(enable_debug + "_10.png", debug_image_mask)
        cv2ext.secure_write(enable_debug + "_11.png", img_mask_erode)

    return img_mask_erode


def remove_points_inside_images_in_contours(
    contours: Any,
    image: Any,
    param: FindImageParameters,
    page_angle: Optional[float],
    enable_debug: Optional[str],
) -> List[Any]:
    mask_with_images = find_images(image, param, page_angle, enable_debug)

    contours_filtered = []
    for contour_i in contours:
        # keep point only outside of images found with find_images.
        csi = list(
            filter(
                lambda point_ij: mask_with_images[
                    point_ij[0][1], point_ij[0][0]
                ]
                == 0,
                contour_i,
            )
        )
        if len(csi) == 0:
            continue
        csi_array = np.ndarray((len(csi), 1, 2), dtype=np.int32)
        csi_array[:, 0, :] = csi[:][:]
        contours_filtered.append(csi_array)
        # Keep only the 3 biggest contours :
        # Two in case of one contour for each page.
        # The third in just for debug.
        if len(contours_filtered) == 3:
            break
    return contours_filtered
