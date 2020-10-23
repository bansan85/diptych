from typing import Any, Dict, Optional, Union, Tuple

import cv2
import numpy as np

from print_release import PrintRelease
from print_test import PrintTest
from print_interface import PrintInterface, ConstString
from page.crop import CropAroundDataInPageParameters
from page.split import (
    FoundSplitLineWithLineParameters,
    FoundSplitLineWithWave,
    SplitTwoWavesParameters,
)
import page.split
import page.unskew
from page.unskew import UnskewPageParameters
import cv2ext
import pages
import compute


class SeparatePage:
    # Sépare une page en deux en détectant la vague dans le papier en haut et
    # en bas de la reliure.
    def split_two_waves(
        self,
        image: Any,
        parameters: SplitTwoWavesParameters,
        enable_debug: Optional[str],
    ) -> Tuple[Any, Any]:
        param1 = FoundSplitLineWithLineParameters(
            parameters.blur_size,
            parameters.thresholds_min,
            parameters.erode,
            parameters.canny,
            parameters.hough_lines,
            parameters.delta_rho,
            parameters.delta_tetha,
        )
        first = page.split.found_split_line_with_line(
            image, param1, compute.optional_concat(enable_debug, "_1")
        )

        param2 = FoundSplitLineWithWave(
            parameters.blur_size,
            parameters.threshold,
            parameters.erode,
            parameters.find_images,
            parameters.find_candidates,
        )
        second = page.split.found_split_line_with_wave(
            image, param2, compute.optional_concat(enable_debug, "_2")
        )

        angle_moy, pos_moy = page.split.find_best_split_in_all_candidates(
            first, second
        )

        self.__output.print(
            ConstString.separation_double_page_angle(), angle_moy
        )
        self.__output.print(ConstString.separation_double_page_y(), pos_moy)

        page_gauche, page_droite = cv2ext.split_image(
            image, angle_moy, pos_moy
        )

        cv2ext.write_image_if(page_gauche, enable_debug, "_3_1.png")
        cv2ext.write_image_if(page_droite, enable_debug, "_3_2.png")

        # On renvoie les images cropées.
        return page_gauche, page_droite

    def unskew_page(
        self,
        image: Any,
        n_page: int,
        parameters: UnskewPageParameters,
        enable_debug: Optional[str],
    ) -> Any:
        rotate_angle = page.unskew.find_rotation(
            image, n_page, parameters, enable_debug
        )

        self.__output.print(ConstString.page_rotation(n_page), rotate_angle)
        # Enfin, on tourne.
        img_rotated = cv2ext.rotate_image(image, rotate_angle)
        if enable_debug is not None:
            cv2.imwrite(
                enable_debug + "_" + str(n_page) + "_5.png", img_rotated
            )
        return img_rotated

    def crop_around_data_in_page(
        self,
        image: Any,
        n_page: int,
        parameters: CropAroundDataInPageParameters,
        enable_debug: Optional[str],
    ) -> Tuple[Any, Tuple[int, int, int, int], int, int]:
        crop_rect_size = page.crop.crop_around_page(
            image, n_page, parameters, enable_debug
        )

        page_gauche_0 = cv2ext.crop_rectangle(image, crop_rect_size)
        cv2ext.write_image_if(
            page_gauche_0, enable_debug, "_" + str(n_page) + "_6.png"
        )

        crop_rect2_size = page.crop.crop_around_data(
            page_gauche_0, n_page, parameters, enable_debug
        )

        imgh, imgw = cv2ext.get_hw(page_gauche_0)
        # Aucun contour, page vide : on renvoie une image d'un pixel
        if crop_rect2_size is None:
            self.__output.print(
                ConstString.image_crop(n_page, "x1"), int(imgw / 2) - 1
            )
            self.__output.print(
                ConstString.image_crop(n_page, "y1"), int(imgh / 2) - 1
            )
            self.__output.print(
                ConstString.image_crop(n_page, "x2"), int(imgw / 2)
            )
            self.__output.print(
                ConstString.image_crop(n_page, "y2"), int(imgh / 2)
            )

            return (
                page_gauche_0,
                (
                    int(imgw / 2) - 1,
                    int(imgw / 2),
                    int(imgh / 2) - 1,
                    int(imgh / 2),
                ),
                imgw,
                imgh,
            )

        self.__output.print(
            ConstString.image_crop(n_page, "x1"),
            crop_rect_size[0] + crop_rect2_size[0],
        )
        self.__output.print(
            ConstString.image_crop(n_page, "y1"),
            crop_rect_size[2] + crop_rect2_size[2],
        )
        self.__output.print(
            ConstString.image_crop(n_page, "x2"),
            crop_rect_size[0] + crop_rect2_size[1],
        )
        self.__output.print(
            ConstString.image_crop(n_page, "y2"),
            crop_rect_size[2] + crop_rect2_size[3],
        )

        return (
            page_gauche_0,
            (
                np.maximum(crop_rect2_size[0] - parameters.border, 0),
                np.minimum(crop_rect2_size[1] + parameters.border, imgw - 1),
                np.maximum(crop_rect2_size[2] - parameters.border, 0),
                np.minimum(crop_rect2_size[3] + parameters.border, imgh - 1),
            ),
            imgw,
            imgh,
        )

    def treat_file(
        self,
        filename: str,
        dict_test: Optional[Dict[str, Any]] = None,
        dict_default_values: Optional[
            Dict[str, Union[int, float, Tuple[int, int]]]
        ] = None,
        enable_debug: bool = False,
    ) -> None:
        print(filename)
        img = cv2ext.charge_image(filename)
        if img is None:
            raise Exception("Failed to load image.", filename)

        if dict_test is None:
            self.__output = PrintRelease()
        else:
            self.__output = PrintTest(dict_test)

        parameters = pages.init_default_values(dict_default_values)

        image1, image2 = self.split_two_waves(
            img,
            parameters.split_two_waves,
            compute.optional_str(enable_debug, filename),
        )

        image1a = self.unskew_page(
            image1,
            1,
            parameters.unskew_page,
            compute.optional_str(enable_debug, filename + "_4"),
        )
        image2a = self.unskew_page(
            image2,
            2,
            parameters.unskew_page,
            compute.optional_str(enable_debug, filename + "_4"),
        )

        (image1a2, crop1, imgw1, imgh1,) = self.crop_around_data_in_page(
            image1a,
            1,
            parameters.crop_around_data_in_page,
            compute.optional_str(enable_debug, filename + "_5"),
        )
        image1b = cv2ext.crop_rectangle(image1a2, crop1)
        (image2a2, crop2, imgw2, imgh2,) = self.crop_around_data_in_page(
            image2a,
            2,
            parameters.crop_around_data_in_page,
            compute.optional_str(enable_debug, filename + "_5"),
        )
        image2b = cv2ext.crop_rectangle(image2a2, crop2)

        if enable_debug:
            cv2.imwrite(filename + "_5_1_10.png", image1b)
            cv2.imwrite(filename + "_5_2_10.png", image2b)

        dpi = compute.find_dpi(imgw1, imgh1, 21.0, 29.7)
        self.__output.print(ConstString.image_dpi(1), dpi)
        recadre1 = cv2ext.add_border_to_match_size(
            image1b,
            (21.0, 29.7),
            crop1,
            (imgw1, imgh1),
            dpi,
        )
        self.__output.print(
            ConstString.image_border(1, 1),
            recadre1[0],
        )
        self.__output.print(
            ConstString.image_border(1, 2),
            recadre1[1],
        )
        self.__output.print(
            ConstString.image_border(1, 3),
            recadre1[2],
        )
        self.__output.print(
            ConstString.image_border(1, 4),
            recadre1[3],
        )
        image1c = cv2.copyMakeBorder(
            image1b,
            recadre1[0],
            recadre1[1],
            recadre1[2],
            recadre1[3],
            cv2.BORDER_CONSTANT,
            value=[255, 255, 255],
        )

        dpi = compute.find_dpi(imgw2, imgh2, 21.0, 29.7)
        self.__output.print(ConstString.image_dpi(2), dpi)
        recadre2 = cv2ext.add_border_to_match_size(
            image2b,
            (21.0, 29.7),
            crop2,
            (imgw2, imgh2),
            dpi,
        )
        self.__output.print(
            ConstString.image_border(2, 1),
            recadre2[0],
        )
        self.__output.print(
            ConstString.image_border(2, 2),
            recadre2[1],
        )
        self.__output.print(
            ConstString.image_border(2, 3),
            recadre2[2],
        )
        self.__output.print(
            ConstString.image_border(2, 4),
            recadre2[3],
        )
        image2c = cv2.copyMakeBorder(
            image2b,
            recadre2[0],
            recadre2[1],
            recadre2[2],
            recadre2[3],
            cv2.BORDER_CONSTANT,
            value=[255, 255, 255],
        )

        cv2.imwrite(filename + "_page_1.png", image1c)
        cv2.imwrite(filename + "_page_2.png", image2c)

        self.__output.close()

    __output: PrintInterface
