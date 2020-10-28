import types
from typing import Union, Tuple, Any, Optional, Iterable, List

import numpy as np
import cv2

from parameters import ErodeParameters, CannyParameters, HoughLinesParameters
from page.find_images import FindImageParameters
import page.find_images
import cv2ext
import compute


class FoundSplitLineWithLineParameters:
    class Impl(types.SimpleNamespace):
        blur_size: Tuple[int, int]
        erode: ErodeParameters
        canny: CannyParameters
        hough_lines: HoughLinesParameters
        limit_rho: int
        limit_tetha: float

    def __init__(  # pylint: disable=too-many-arguments
        self,
        blur_size: Tuple[int, int],
        erode: ErodeParameters,
        canny: CannyParameters,
        hough_lines: HoughLinesParameters,
        limit_rho: int,
        limit_tetha: float,
    ):
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
    def limit_tetha(self) -> float:
        return self.__param.limit_tetha

    @limit_tetha.setter
    def limit_tetha(self, val: float) -> None:
        self.__param.limit_tetha = val


class FindCandidatesSplitLineWithWaveParameters:
    class Impl(types.SimpleNamespace):
        rapport_rect1_rect2: float
        found_contour_iterations: int
        wave_top: float
        wave_bottom: float
        wave_left: float
        wave_right: float

    def __init__(  # pylint: disable=too-many-arguments
        self,
        rapport_rect1_rect2: float,
        found_contour_iterations: int,
        wave_top: float,
        wave_bottom: float,
        wave_left: float,
        wave_right: float,
    ):
        self.__param = FindCandidatesSplitLineWithWaveParameters.Impl(
            rapport_rect1_rect2=rapport_rect1_rect2,
            found_contour_iterations=found_contour_iterations,
            wave_top=wave_top,
            wave_bottom=wave_bottom,
            wave_left=wave_left,
            wave_right=wave_right,
        )

    @property
    def rapport_rect1_rect2(
        self,
    ) -> float:
        return self.__param.rapport_rect1_rect2

    @rapport_rect1_rect2.setter
    def rapport_rect1_rect2(self, val: float) -> None:
        self.__param.rapport_rect1_rect2 = val

    @property
    def found_contour_iterations(
        self,
    ) -> int:
        return self.__param.found_contour_iterations

    @found_contour_iterations.setter
    def found_contour_iterations(self, val: int) -> None:
        self.__param.found_contour_iterations = val

    @property
    def wave_top(self) -> float:
        return self.__param.wave_top

    @wave_top.setter
    def wave_top(self, val: float) -> None:
        self.__param.wave_top = val

    @property
    def wave_bottom(self) -> float:
        return self.__param.wave_bottom

    @wave_bottom.setter
    def wave_bottom(self, val: float) -> None:
        self.__param.wave_bottom = val

    @property
    def wave_left(self) -> float:
        return self.__param.wave_left

    @wave_left.setter
    def wave_left(self, val: float) -> None:
        self.__param.wave_left = val

    @property
    def wave_right(self) -> float:
        return self.__param.wave_right

    @wave_right.setter
    def wave_right(self, val: float) -> None:
        self.__param.wave_right = val

    def init_default_values(
        self,
        key: str,
        value: Union[int, float, Tuple[int, int]],
    ) -> None:
        if key == "RapportRect1Rect2" and isinstance(value, float):
            self.rapport_rect1_rect2 = value
        elif key == "FoundContourIterations" and isinstance(value, int):
            self.found_contour_iterations = value
        elif key == "Top" and isinstance(value, float):
            self.wave_top = value
        elif key == "Bottom" and isinstance(value, float):
            self.wave_bottom = value
        elif key == "Left" and isinstance(value, float):
            self.wave_left = value
        elif key == "Right" and isinstance(value, float):
            self.wave_right = value
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
    ):
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
        limit_tetha: float

    def __init__(  # pylint: disable=too-many-arguments
        self,
        blur_size: Tuple[int, int],
        canny: CannyParameters,
        hough_lines: HoughLinesParameters,
        erode: ErodeParameters,
        limit_rho: int,
        limit_tetha: float,
    ):
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
    ) -> float:
        return self.__param.limit_tetha

    @limit_tetha.setter
    def limit_tetha(self, val: float) -> None:
        self.__param.limit_tetha = val


class SplitTwoWavesParameters:
    class Impl(types.SimpleNamespace):
        erode: ErodeParameters = ErodeParameters((3, 3), 10)
        blur_size: Tuple[int, int] = (10, 10)
        canny: CannyParameters = CannyParameters(25, 255, 5)
        hough_lines: HoughLinesParameters = HoughLinesParameters(
            1, np.pi / (180 * 40), 150, 200, 150
        )
        delta_rho: int = 200
        delta_tetha: float = 20.0
        find_images: FindImageParameters = FindImageParameters(
            0.02, (10, 10), (10, 10), (10, 10), 8, 0.01
        )
        find_candidates: FindCandidatesSplitLineWithWaveParameters = (
            FindCandidatesSplitLineWithWaveParameters(
                1.05,
                10,
                0.2,
                0.8,
                0.4,
                0.6,
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
    def delta_tetha(self) -> float:
        return self.__param.delta_tetha

    @delta_tetha.setter
    def delta_tetha(self, val: float) -> None:
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
        value: Union[int, float, Tuple[int, int]],
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
        elif key == "DeltaTetha" and isinstance(value, float):
            self.delta_tetha = value
        elif key.startswith("Wave"):
            self.find_candidates.init_default_values(key[len("Wave") :], value)
        else:
            raise Exception("Invalid property.", key)


def __found_candidates_split_line_with_line(
    image: Any,
    param: FindCandidatesSplitLineWithLineParameters,
    enable_debug: Optional[str] = None,
) -> Iterable[Tuple[int, int, int, int]]:
    blurimg = cv2ext.force_image_to_be_grayscale(image, param.blur_size, False)
    cv2ext.write_image_if(blurimg, enable_debug, "_2_2.png")
    blurimg_equ = cv2.equalizeHist(blurimg)
    cv2ext.write_image_if(blurimg_equ, enable_debug, "_2_2b.png")
    _, threshold = cv2.threshold(
        blurimg_equ,
        cv2ext.threshold_from_gaussian_histogram(blurimg_equ),
        255,
        cv2.THRESH_BINARY,
    )
    cv2ext.write_image_if(threshold, enable_debug, "_2_3.png")
    erode_dilate = cv2ext.erode_and_dilate(
        threshold, param.erode.size, param.erode.iterations
    )
    cv2ext.write_image_if(erode_dilate, enable_debug, "_2_4.png")
    canny = cv2.Canny(
        erode_dilate,
        param.canny.minimum,
        param.canny.maximum,
        apertureSize=param.canny.aperture_size,
    )
    cv2ext.write_image_if(canny, enable_debug, "_2_5.png")
    list_lines_p = cv2.HoughLinesP(
        canny,
        param.hough_lines.delta_rho,
        param.hough_lines.delta_tetha,
        param.hough_lines.threshold,
        minLineLength=param.hough_lines.min_line_length,
        maxLineGap=param.hough_lines.max_line_gap,
    )
    if enable_debug is not None:
        cv2.imwrite(
            enable_debug + "_2_6.png",
            cv2ext.draw_lines_from_hough_lines(
                image, list_lines_p, (0, 0, 255), 1
            ),
        )
    list_lines = map(lambda p: p[0], list_lines_p)

    # keep vertical lines and in the center of the image
    return filter(
        lambda x: compute.keep_angle_pos_closed_to_target(
            x,
            param.limit_tetha,
            90,
            cv2ext.get_hw(blurimg)[1] // 2,
            param.limit_rho,
        ),
        list_lines,
    )


def __loop_to_find_best_mean_angle_pos(
    histogram_posx: Any,
    histogram_posx_maxy: Any,
    histogram_posx_miny: Any,
    size: Tuple[int, int],
    valid_lines: List[Tuple[int, int, int, int]],
) -> Tuple[float, int]:
    width, height = size

    old_angle_i = 0.0
    n_same_angle = 0
    posx = 0
    for __size__ in range(1, 100, 2):
        histogram_posx_blur = cv2.GaussianBlur(
            histogram_posx,
            (1, __size__),
            __size__,
            borderType=cv2.BORDER_REPLICATE,
        )
        histogram_list_of_top = np.zeros(width + 1)
        list_of_top = compute.get_tops_indices_histogram(histogram_posx_blur)
        top_found = False
        for i in list_of_top:
            length_i = max(
                histogram_posx_maxy[i - __size__ : i + __size__]
            ) - min(histogram_posx_miny[i - __size__ : i + __size__])
            if length_i > 0.5 * height:
                top_found = True
                histogram_list_of_top[i] = sum(
                    filter(
                        lambda x: x > 0,
                        histogram_posx_maxy[i - __size__ : i + __size__]
                        - histogram_posx_miny[i - __size__ : i + __size__],
                    )
                )
        if not top_found:
            continue
        histogram_list_of_top_blur = cv2.GaussianBlur(
            histogram_list_of_top,
            (1, __size__),
            __size__,
            borderType=cv2.BORDER_REPLICATE,
        )
        posx = np.argmax(histogram_list_of_top_blur)

        count = 0
        anglex = 0.0

        all_angle_pos = map(
            lambda x: compute.get_angle_0_180_posx((x[0], x[1]), (x[2], x[3])),
            valid_lines,
        )
        valid_angle_pos = filter(
            lambda x: x[1] is not None
            # pylint: disable=cell-var-from-loop
            and x[1] - __size__ <= posx <= x[1] + __size__,
            all_angle_pos,
        )

        for angle, _ in valid_angle_pos:
            anglex = anglex + angle
            count = count + 1

        if count != 0:
            if anglex / count == old_angle_i:
                n_same_angle = n_same_angle + 1
            else:
                n_same_angle = 1
            old_angle_i = anglex / count

        if n_same_angle == 3:
            break

    return (old_angle_i, posx)


def __best_candidates_split_line_with_line(
    valid_lines: List[Tuple[int, int, int, int]],
    width: int,
    height: int,
) -> Tuple[float, int]:
    histogram_posx = np.zeros(width + 1)
    histogram_posx_miny = height * np.ones(width + 1)
    histogram_posx_maxy = np.zeros(width + 1)
    for point1_x, point1_y, point2_x, point2_y in valid_lines:
        _, pos = compute.get_angle_0_180_posx(
            (point1_x, point1_y), (point2_x, point2_y)
        )
        length = np.linalg.norm(
            np.array((point1_x, point1_y)) - np.array((point2_x, point2_y))
        )
        histogram_posx[pos] = histogram_posx[pos] + length
        histogram_posx_miny[pos] = min(
            histogram_posx_miny[pos], point1_y, point2_y
        )
        histogram_posx_maxy[pos] = max(
            histogram_posx_maxy[pos], point1_y, point2_y
        )

    return __loop_to_find_best_mean_angle_pos(
        histogram_posx,
        histogram_posx_maxy,
        histogram_posx_miny,
        (width, height),
        valid_lines,
    )


def found_split_line_with_line(
    image: Any,
    param: FoundSplitLineWithLineParameters,
    enable_debug: Optional[str] = None,
) -> Tuple[float, int]:
    cv2ext.write_image_if(image, enable_debug, "_1.png")

    valid_lines: List[Tuple[int, int, int, int]] = []

    valid_lines.extend(
        __found_candidates_split_line_with_line(
            image,
            FindCandidatesSplitLineWithLineParameters(
                param.blur_size,
                param.canny,
                param.hough_lines,
                param.erode,
                param.limit_rho,
                param.limit_tetha,
            ),
            enable_debug,
        )
    )

    if len(valid_lines) == 0:
        raise Exception(
            "found_split_line_with_line",
            "Failed to find candidates for the separator line.",
        )

    (angle_1, posx_1,) = __best_candidates_split_line_with_line(
        valid_lines,
        cv2ext.get_hw(image)[1],
        cv2ext.get_hw(image)[0],
    )

    height, _ = cv2ext.get_hw(image)
    point_1a = (posx_1, 0)
    point_1b = (
        int(posx_1 - np.tan((angle_1 - 90.0) / 180.0 * np.pi) * height),
        height - 1,
    )

    if enable_debug is not None:
        image_with_lines = cv2ext.convertion_en_couleur(image)
        cv2.line(
            image_with_lines,
            (point_1a[0], point_1a[1]),
            (point_1b[0], point_1b[1]),
            (255, 0, 0),
            5,
        )
        for line in valid_lines:
            cv2.line(
                image_with_lines,
                (line[0], line[1]),
                (line[2], line[3]),
                (0, 0, 255),
                1,
            )
        cv2.imwrite(enable_debug + "_7.png", image_with_lines)

    return angle_1, posx_1


def __found_candidates_split_line_with_wave_keep_interesting_points(
    polygon: Any,
    toppointsi: List[Tuple[int, int]],
    bottompointsi: List[Tuple[int, int]],
    param: FindCandidatesSplitLineWithWaveParameters,
    image: Any,
) -> None:
    height, width = cv2ext.get_hw(image)
    for point_i in polygon:
        point_x, point_y = point_i[0]
        if param.wave_left * width < point_x < param.wave_right * width:
            if point_y < param.wave_top * height:
                toppointsi.append((point_x, point_y))
            elif point_y > param.wave_bottom * height:
                bottompointsi.append((point_x, point_y))


def __found_candidates_split_line_with_wave(
    contour: Any,
    image: Any,
    eroded: Any,
    param: FindCandidatesSplitLineWithWaveParameters,
    enable_debug: Optional[str] = None,
) -> Tuple[Any, Any]:
    nb_rectangle = (
        int(
            cv2.contourArea(contour[0]) / cv2.contourArea(contour[1])
            < param.rapport_rect1_rect2
        )
        + 1
    )
    if enable_debug is not None:
        img_tmp = cv2.cvtColor(eroded, cv2.COLOR_GRAY2BGR)
        for i in range(nb_rectangle):
            img_tmp = cv2.drawContours(img_tmp, contour, i, (255, 0, 0), 10)
        img_tmp = cv2.drawContours(
            img_tmp, contour, nb_rectangle, (0, 0, 255), 3
        )
        cv2.imwrite(enable_debug + "_5.png", img_tmp)

    # Il faut au minimum 10 points pour détecter le décroché qui indique
    # la séparation entre deux pages.
    if nb_rectangle == 1:
        list_of_contours = [contour[0]]
        # Si on a un seul contour pour la double page, la vague peut être
        # caractérisée par 3 points.
        nb_point_in_wave = 3
    else:
        list_of_contours = [contour[0], contour[1]]
        # Si on a deux contours pour la double page, la vague ne peut être
        # caractérisée que par 2 points.
        nb_point_in_wave = 2
    toppoints = []
    bottompoints = []
    for contour_i in list_of_contours:
        npoints = 6
        for nloop in range(param.found_contour_iterations):
            polygon = cv2ext.get_polygon_from_contour(contour_i, npoints)
            if enable_debug is not None:
                img_tmp = cv2.drawContours(
                    cv2ext.convertion_en_couleur(image),
                    [polygon],
                    0,
                    (0, 0, 255),
                    10,
                )
                for i in range(nb_rectangle):
                    cv2.drawContours(img_tmp, contour, i, (255, 0, 0), 10)
                cv2.imwrite(
                    enable_debug + "_6_" + str(nloop) + ".png", img_tmp
                )
            toppointsi: List[Tuple[int, int]] = []
            bottompointsi: List[Tuple[int, int]] = []
            __found_candidates_split_line_with_wave_keep_interesting_points(
                polygon, toppointsi, bottompointsi, param, image
            )
            # On filtre les points qui sont à peu près au centre (x/2) de
            # l'image
            # Est-ce qu'on a suffisamment de points ?
            if (
                len(toppointsi) >= nb_point_in_wave
                and len(bottompointsi) >= nb_point_in_wave
            ):
                toppoints.append(toppointsi)
                bottompoints.append(bottompointsi)
                break
            npoints = len(polygon) + 1

    return (toppoints, bottompoints)


def found_split_line_with_wave(
    image: Any,
    parameters: FoundSplitLineWithWave,
    enable_debug: Optional[str] = None,
) -> Tuple[float, int]:
    cv2ext.write_image_if(image, enable_debug, "_1.png")
    blurimg = cv2ext.force_image_to_be_grayscale(
        image, parameters.blur_size, False
    )
    cv2ext.write_image_if(blurimg, enable_debug, "_2.png")
    blurimg_equ = cv2.equalizeHist(blurimg)
    cv2ext.write_image_if(blurimg_equ, enable_debug, "_2b.png")
    _, threshold = cv2.threshold(
        blurimg_equ,
        cv2ext.threshold_from_gaussian_histogram(blurimg_equ),
        255,
        cv2.THRESH_BINARY,
    )
    cv2ext.write_image_if(threshold, enable_debug, "_3.png")
    erode_dilate = cv2ext.erode_and_dilate(
        threshold, parameters.erode.size, parameters.erode.iterations
    )
    cv2ext.write_image_if(erode_dilate, enable_debug, "_4.png")
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
    cv2ext.write_image_if(eroded_bordered, enable_debug, "_4a.png")

    # On cherche tous les contours
    contours, _ = cv2.findContours(
        eroded_bordered, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
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
    if enable_debug is not None:
        img_tmp = cv2.cvtColor(eroded_bordered, cv2.COLOR_GRAY2BGR)
        img_tmp = cv2.drawContours(
            img_tmp, sorted_contours, 0, (255, 0, 0), 10
        )
        img_tmp = cv2.drawContours(
            img_tmp, sorted_contours, 1, (0, 255, 0), 10
        )
        cv2.imwrite(enable_debug + "_4b.png", img_tmp)

    cv2ext.remove_border_in_contours(sorted_contours, size_border, image)

    cs2 = page.find_images.remove_points_inside_images_in_contours(
        sorted_contours,
        image,
        parameters.find_images,
        compute.optional_concat(enable_debug, "_4c_wave"),
    )

    toppoints, bottompoints = __found_candidates_split_line_with_wave(
        cs2,
        image,
        erode_dilate,
        parameters.find_candidates,
        enable_debug,
    )

    if len(toppoints) == 0 or len(bottompoints) == 0:
        raise Exception(
            "Failed to found contour of the page.",
            str(parameters.find_candidates.found_contour_iterations)
            + " iterations",
        )

    # Le point de séparation des deux pages en haut et bas
    maxtoppoints = list(map(lambda x: max(x, key=lambda x: x[1]), toppoints))
    minbottompoints = list(
        map(lambda x: min(x, key=lambda x: x[1]), bottompoints)
    )
    toppoint = (
        sum(x for x, y in maxtoppoints) // len(maxtoppoints),
        sum(y for x, y in maxtoppoints) // len(maxtoppoints),
    )
    bottompoint = (
        sum(x for x, y in minbottompoints) // len(minbottompoints),
        sum(y for x, y in minbottompoints) // len(minbottompoints),
    )
    if enable_debug is not None:
        image_with_lines = cv2ext.convertion_en_couleur(image)
        cv2.line(
            image_with_lines,
            toppoint,
            bottompoint,
            (0, 0, 255),
            5,
        )
        cv2.imwrite(enable_debug + "_7.png", image_with_lines)
    angle_ret, posx_ret = compute.get_angle_0_180_posx(bottompoint, toppoint)
    if posx_ret is None:
        raise Exception("Failed to found vertical line.")
    return angle_ret, posx_ret


def find_best_split_in_all_candidates(
    one: Tuple[float, int], two: Tuple[float, int]
) -> Tuple[float, int]:
    angle_moy = (one[0] + two[0]) / 2
    pos_moy = (one[1] + two[1]) // 2
    return (angle_moy, pos_moy)
