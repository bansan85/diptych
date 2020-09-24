import numpy as np


class RectangleContourParameters:
    def __init__(self, min_e, max_e, max_iterations):
        self.__min_e = min_e
        self.__max_e = max_e
        self.__max_iterations = max_iterations

    @property
    def MinE(self):
        return self.__min_e

    @MinE.setter
    def MinE(self, val):
        self.__min_e = val

    @property
    def MaxE(self):
        return self.__max_e

    @MaxE.setter
    def MaxE(self, val):
        self.__max_e = val

    @property
    def MaxIterations(self):
        return self.__max_iterations

    @MaxIterations.setter
    def MaxIterations(self, val):
        self.__max_iterations = val


class ErodeParameters:
    def __init__(self, size, iterations):
        self.__size = size
        self.__iterations = iterations

    @property
    def Size(self):
        return self.__size

    @Size.setter
    def Size(self, val):
        self.__size = val

    @property
    def Iterations(self):
        return self.__iterations

    @Iterations.setter
    def Iterations(self, val):
        self.__iterations = val


class CannyParameters:
    def __init__(self, mini, maxi, aperture_size):
        self.__min = mini
        self.__max = maxi
        self.__aperture_size = aperture_size

    @property
    def Min(self):
        return self.__min

    @Min.setter
    def Min(self, val):
        self.__min = val

    @property
    def Max(self):
        return self.__max

    @Max.setter
    def Max(self, val):
        self.__max = val

    @property
    def ApertureSize(self):
        return self.__aperture_size

    @ApertureSize.setter
    def ApertureSize(self, val):
        self.__aperture_size = val


class DeltaRhoTethaParameters:
    def __init__(self, rho, tetha):
        self.__rho = rho
        self.__tetha = tetha

    @property
    def Rho(self):
        return self.__rho

    @Rho.setter
    def Rho(self, val):
        self.__rho = val

    @property
    def Tetha(self):
        return self.__tetha

    @Tetha.setter
    def Tetha(self, val):
        self.__tetha = val


class HoughLinesParameters:
    def __init__(
        self, delta_rho, delta_tetha, threshold, min_line_length, max_line_gap
    ):
        self.__delta_rho = delta_rho
        self.__delta_tetha = delta_tetha
        self.__threshold = threshold
        self.__min_line_length = min_line_length
        self.__max_line_gap = max_line_gap

    @property
    def DeltaRho(self):
        return self.__delta_rho

    @DeltaRho.setter
    def DeltaRho(self, val):
        self.__delta_rho = val

    @property
    def DeltaTetha(self):
        return self.__delta_tetha

    @DeltaTetha.setter
    def DeltaTetha(self, val):
        self.__delta_tetha = val

    @property
    def Threshold(self):
        return self.__threshold

    @Threshold.setter
    def Threshold(self, val):
        self.__threshold = val

    @property
    def MinLineLength(self):
        return self.__min_line_length

    @MinLineLength.setter
    def MinLineLength(self, val):
        self.__min_line_length = val

    @property
    def MaxLineGap(self):
        return self.__max_line_gap

    @MaxLineGap.setter
    def MaxLineGap(self, val):
        self.__max_line_gap = val


class ThresholdParameters:
    def __init__(self, mini, maxi):
        self.__min = mini
        self.__max = maxi

    @property
    def Min(self):
        return self.__min

    @Min.setter
    def Min(self, val):
        self.__min = val

    @property
    def Max(self):
        return self.__max

    @Max.setter
    def Max(self, val):
        self.__max = val


class FoundSplitLineWithLineParameters:
    def __init__(self, blur_size, threshold, erode, canny, hough_lines, limit):
        self.__blur_size = blur_size
        self.__threshold = threshold
        self.__erode = erode
        self.__canny = canny
        self.__hough_lines = hough_lines
        self.__limit = limit

    @property
    def BlurSize(self):
        return self.__blur_size

    @property
    def Threshold(self):
        return self.__threshold

    @property
    def Erode(self):
        return self.__erode

    @property
    def Canny(self):
        return self.__canny

    @property
    def HoughLines(self):
        return self.__hough_lines

    @property
    def Limit(self):
        return self.__limit


class FoundSplitLineWithWave:
    def __init__(
        self,
        blur_size,
        threshold,
        erode,
        rapport_rect1_rect2,
        npoints_2pages,
        npoints_1page,
        found_contour_iterations,
        rectangle_contour,
        wave_top,
        wave_bottom,
        wave_left,
        wave_right,
    ):
        self.__blur_size = blur_size
        self.__threshold = threshold
        self.__erode = erode
        self.__rapport_rect1_rect2 = rapport_rect1_rect2
        self.__npoints_2pages = npoints_2pages
        self.__npoints_1page = npoints_1page
        self.__found_contour_iterations = found_contour_iterations
        self.__rectangle_contour = rectangle_contour
        self.__wave_top = wave_top
        self.__wave_bottom = wave_bottom
        self.__wave_left = wave_left
        self.__wave_right = wave_right

    @property
    def BlurSize(self):
        return self.__blur_size

    @BlurSize.setter
    def BlurSize(self, val):
        self.__blur_size = val

    @property
    def Threshold(self):
        return self.__threshold

    @property
    def Erode(self):
        return self.__erode

    @property
    def RapportRect1Rect2(self):
        return self.__rapport_rect1_rect2

    @RapportRect1Rect2.setter
    def RapportRect1Rect2(self, val):
        self.__rapport_rect1_rect2 = val

    @property
    def Npoints2pages(self):
        return self.__npoints_2pages

    @Npoints2pages.setter
    def Npoints2pages(self, val):
        self.__npoints_2pages = val

    @property
    def Npoints1page(self):
        return self.__npoints_1page

    @Npoints1page.setter
    def Npoints1page(self, val):
        self.__npoints_1page = val

    @property
    def FoundContourIterations(self):
        return self.__found_contour_iterations

    @FoundContourIterations.setter
    def FoundContourIterations(self, val):
        self.__found_contour_iterations = val

    @property
    def RectangleContour(self):
        return self.__rectangle_contour

    @property
    def WaveTop(self):
        return self.__wave_top

    @WaveTop.setter
    def WaveTop(self, val):
        self.__wave_top = val

    @property
    def WaveBottom(self):
        return self.__wave_bottom

    @WaveBottom.setter
    def WaveBottom(self, val):
        self.__wave_bottom = val

    @property
    def WaveLeft(self):
        return self.__wave_left

    @WaveLeft.setter
    def WaveLeft(self, val):
        self.__wave_left = val

    @property
    def WaveRight(self):
        return self.__wave_right

    @WaveRight.setter
    def WaveRight(self, val):
        self.__wave_right = val


class SplitTwoWavesParameters:
    def __init__(self):
        self.__erode = ErodeParameters((4, 4), 1)
        self.__blur_size = (10, 10)
        self.__threshold = ThresholdParameters(240, 255)
        self.__canny = CannyParameters(25, 255, 5)
        self.__hough_lines = HoughLinesParameters(1, np.pi / (180 * 20), 50, 300, 30)
        self.__delta_rho_tetha = DeltaRhoTethaParameters(200, 20)
        self.__rapport_rect1_rect2 = 1.05
        self.__wave_top = 0.2
        self.__wave_bottom = 0.8
        self.__wave_left = 0.4
        self.__wave_right = 0.6
        self.__found_contour_iterations = 10
        self.__npoints_2pages = 10
        self.__npoints_1page = 6
        self.__rectangle_contour = RectangleContourParameters(0.00001, 0.99, 10)

    @property
    def Erode(self):
        return self.__erode

    @property
    def BlurSize(self):
        return self.__blur_size

    @BlurSize.setter
    def BlurSize(self, val):
        self.__blur_size = val

    @property
    def Threshold(self):
        return self.__threshold

    @property
    def Canny(self):
        return self.__canny

    @property
    def HoughLines(self):
        return self.__hough_lines

    @property
    def DeltaRhoTetha(self):
        return self.__delta_rho_tetha

    @property
    def RapportRect1Rect2(self):
        return self.__rapport_rect1_rect2

    @RapportRect1Rect2.setter
    def RapportRect1Rect2(self, val):
        self.__rapport_rect1_rect2 = val

    @property
    def WaveTop(self):
        return self.__wave_top

    @WaveTop.setter
    def WaveTop(self, val):
        self.__wave_top = val

    @property
    def WaveBottom(self):
        return self.__wave_bottom

    @WaveBottom.setter
    def WaveBottom(self, val):
        self.__wave_bottom = val

    @property
    def WaveRight(self):
        return self.__wave_right

    @WaveRight.setter
    def WaveRight(self, val):
        self.__wave_right = val

    @property
    def WaveLeft(self):
        return self.__wave_left

    @WaveLeft.setter
    def WaveLeft(self, val):
        self.__wave_left = val

    @property
    def FoundContourIterations(self):
        return self.__found_contour_iterations

    @FoundContourIterations.setter
    def FoundContourIterations(self, val):
        self.__found_contour_iterations = val

    @property
    def Npoints2pages(self):
        return self.__npoints_2pages

    @Npoints2pages.setter
    def Npoints2pages(self, val):
        self.__npoints_2pages = val

    @property
    def Npoints1page(self):
        return self.__npoints_1page

    @Npoints1page.setter
    def Npoints1page(self, val):
        self.__npoints_1page = val

    @property
    def RectangleContour(self):
        return self.__rectangle_contour


class UnskewPageParameters:
    def __init__(self):
        self.__erode = ErodeParameters((2, 2), 7)
        self.__canny = CannyParameters(25, 225, 5)
        self.__hough_lines = HoughLinesParameters(1, np.pi / (180 * 20), 70, 300, 90)

        self.__angle_limit = 20
        self.__angle_limit_stddev = 0.5

    @property
    def Erode(self):
        return self.__erode

    @property
    def Canny(self):
        return self.__canny

    @property
    def HoughLines(self):
        return self.__hough_lines

    @property
    def AngleLimit(self):
        return self.__angle_limit

    @AngleLimit.setter
    def AngleLimit(self, val):
        self.__angle_limit = val

    @property
    def AngleLimitStddev(self):
        return self.__angle_limit_stddev

    @AngleLimitStddev.setter
    def AngleLimitStddev(self, val):
        self.__angle_limit_stddev = val


class CropAroundDataInPageParameters:
    def __init__(self):
        self.__erode = ErodeParameters((9, 9), 1)
        self.__threshold1 = ThresholdParameters(240, 255)
        self.__rectangle_contour = RectangleContourParameters(0.00001, 0.99, 10)
        self.__dilate_size = (2, 2)
        self.__threshold2 = ThresholdParameters(200, 255)
        self.__contour_area_min = 0.002 * 0.002
        self.__contour_area_max = 0.5 * 0.5
        self.__border = 10

    @property
    def Erode(self):
        return self.__erode

    @property
    def Threshold1(self):
        return self.__threshold1

    @property
    def RectangleContour(self):
        return self.__rectangle_contour

    @property
    def DilateSize(self):
        return self.__dilate_size

    @DilateSize.setter
    def DilateSize(self, val):
        self.__dilate_size = val

    @property
    def Threshold2(self):
        return self.__threshold2

    @property
    def ContourAreaMin(self):
        return self.__contour_area_min

    @ContourAreaMin.setter
    def ContourAreaMin(self, val):
        self.__contour_area_min = val

    @property
    def ContourAreaMax(self):
        return self.__contour_area_max

    @ContourAreaMax.setter
    def ContourAreaMax(self, val):
        self.__contour_area_max = val

    @property
    def Border(self):
        return self.__border

    @Border.setter
    def Border(self, val):
        self.__border = val


class Parameters:
    def __init__(self):
        self.split_two_waves = SplitTwoWavesParameters()
        self.unskew_page = UnskewPageParameters()
        self.crop_around_data_in_page = CropAroundDataInPageParameters()

    def init_default_values(default_values):
        parameters = Parameters()

        for param, value in default_values.items():
            if param == "SplitTwoWavesErodeSize":
                parameters.split_two_waves.Erode.Size = value
            elif param == "SplitTwoWavesErodeIterations":
                parameters.split_two_waves.Erode.Iterations = value
            elif param == "SplitTwoWavesBlurSize":
                parameters.split_two_waves.BlurSize = value
            elif param == "SplitTwoWavesThresholdMin":
                parameters.split_two_waves.Threshold.Min = value
            elif param == "SplitTwoWavesThresholdMax":
                parameters.split_two_waves.Threshold.Max = value
            elif param == "SplitTwoWavesCannyMin":
                parameters.split_two_waves.Canny.Min = value
            elif param == "SplitTwoWavesCannyMax":
                parameters.split_two_waves.Canny.Max = value
            elif param == "SplitTwoWavesCannyApertureSize":
                parameters.split_two_waves.Canny.ApertureSize = value
            elif param == "SplitTwoWavesHoughLinesDeltaRho":
                parameters.split_two_waves.HoughLines.DeltaRho = value
            elif param == "SplitTwoWavesHoughLinesDeltaTetha":
                parameters.split_two_waves.HoughLines.DeltaTetha = value
            elif param == "SplitTwoWavesHoughLinesThreshold":
                parameters.split_two_waves.HoughLines.Threshold = value
            elif param == "SplitTwoWavesHoughLinesMinLineLength":
                parameters.split_two_waves.HoughLines.MinLineLength = value
            elif param == "SplitTwoWavesHoughLinesMaxLineGap":
                parameters.split_two_waves.HoughLines.MaxLineGap = value
            elif param == "SplitTwoWavesDeltaRhoTethaRho":
                parameters.split_two_waves.DeltaRhoTetha.Rho = value
            elif param == "SplitTwoWavesDeltaRhoTethaTetha":
                parameters.split_two_waves.DeltaRhoTetha.Tetha = value
            elif param == "SplitTwoWavesRapportRect1Rect2":
                parameters.split_two_waves.RapportRect1Rect2 = value
            elif param == "SplitTwoWavesWaveTop":
                parameters.split_two_waves.WaveTop = value
            elif param == "SplitTwoWavesWaveBottom":
                parameters.split_two_waves.WaveBottom = value
            elif param == "SplitTwoWavesWaveLeft":
                parameters.split_two_waves.WaveLeft = value
            elif param == "SplitTwoWavesWaveRight":
                parameters.split_two_waves.WaveRight = value
            elif param == "SplitTwoWavesFoundContourIterations":
                parameters.split_two_waves.FoundContourIterations = value
            elif param == "SplitTwoWavesNpoints2pages":
                parameters.split_two_waves.Npoints2pages = value
            elif param == "SplitTwoWavesNpoints1page":
                parameters.split_two_waves.Npoints1page = value
            elif param == "SplitTwoWavesGetRectangleFromContourMinE":
                parameters.split_two_waves.RectangleContour.MinE = value
            elif param == "SplitTwoWavesGetRectangleFromContourMaxE":
                parameters.split_two_waves.RectangleContour.MaxE = value
            elif param == "SplitTwoWavesGetRectangleFromContourMaxIterations":
                parameters.split_two_waves.RectangleContour.MaxIterations = value

            elif param == "UnskewPageErodeSize":
                parameters.unskew_page.Erode.Size = value
            elif param == "UnskewPageErodeIterations":
                parameters.unskew_page.Erode.Iterations = value
            elif param == "UnskewPageCannyMin":
                parameters.unskew_page.Canny.Min = value
            elif param == "UnskewPageCannyMax":
                parameters.unskew_page.Canny.Max = value
            elif param == "UnskewPageCannyApertureSize":
                parameters.unskew_page.Canny.ApertureSize = value
            elif param == "UnskewPageHoughLinesDeltaRho":
                parameters.unskew_page.HoughLines.DeltaRho = value
            elif param == "UnskewPageHoughLinesDeltaTetha":
                parameters.unskew_page.HoughLines.DeltaTetha = value
            elif param == "UnskewPageHoughLinesThreshold":
                parameters.unskew_page.HoughLines.Threshold = value
            elif param == "UnskewPageHoughLinesMinLineLength":
                parameters.unskew_page.HoughLines.MinLineLength = value
            elif param == "UnskewPageHoughLinesMaxLineGap":
                parameters.unskew_page.HoughLines.MaxLineGap = value
            elif param == "UnskewPageAngleLimit":
                parameters.unskew_page.AngleLimit = value
            elif param == "UnskewPageAngleLimitStddev":
                parameters.unskew_page.AngleLimitStddev = value

            elif param == "CropAroundDataInPageErodeSize":
                parameters.crop_around_data_in_page.Erode.Size = value
            elif param == "CropAroundDataInPageErodeIterations":
                parameters.crop_around_data_in_page.Erode.Iterations = value
            elif param == "CropAroundDataInPageThreshold1Min":
                parameters.crop_around_data_in_page.Threshold1.Min = value
            elif param == "CropAroundDataInPageThreshold1Max":
                parameters.crop_around_data_in_page.Threshold1.Max = value
            elif param == "CropAroundDataInPageGetRectangleFromContourMinE":
                parameters.crop_around_data_in_page.RectangleContour.MinE = value
            elif param == "CropAroundDataInPageGetRectangleFromContourMaxE":
                parameters.crop_around_data_in_page.RectangleContour.MaxE = value
            elif param == "CropAroundDataInPageGetRectangleFromContourMaxIterations":
                parameters.crop_around_data_in_page.RectangleContour.MaxIterations = (
                    value
                )
            elif param == "CropAroundDataInPageDilateSize":
                parameters.crop_around_data_in_page.DilateSize = value
            elif param == "CropAroundDataInPageThreshold2Min":
                parameters.crop_around_data_in_page.Threshold2.Min = value
            elif param == "CropAroundDataInPageThreshold2Max":
                parameters.crop_around_data_in_page.Threshold2.Max = value
            elif param == "CropAroundDataInPageContourAreaMin":
                parameters.crop_around_data_in_page.ContourAreaMin = value
            elif param == "CropAroundDataInPageContourAreaMax":
                parameters.crop_around_data_in_page.ContourAreaMax = value
            elif param == "CropAroundDataInPageBorder":
                parameters.crop_around_data_in_page.Border = value
            else:
                raise Exception("Invalid property.", param)

        return parameters
