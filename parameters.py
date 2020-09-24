import numpy as np

class GetRectangleFromContourParameters:
    def __init__(self):
        self.__min_e = 0.00001
        self.__max_e = 0.99
        self.__max_iterations = 10

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


class SplitTwoWavesParameters:
    def __init__(self):
        self.__erode_size = (4, 4)
        self.__erode_iterations = 1
        self.__blur_size = (10, 10)
        self.__threshold_min = 210
        self.__threshold_max = 255
        self.__rapport_rect1_rect2 = 1.05
        self.__wave_top = 0.2
        self.__wave_bottom = 0.8
        self.__wave_left = 0.4
        self.__wave_right = 0.6
        self.__found_contour_iterations = 10
        self.__npoints_2pages = 10
        self.__npoints_1page = 6
        self.get_rectangle_from_contour = GetRectangleFromContourParameters()

    @property
    def ErodeSize(self):
        return self.__erode_size

    @ErodeSize.setter
    def ErodeSize(self, val):
        self.__erode_size = val

    @property
    def ErodeIterations(self):
        return self.__erode_iterations

    @ErodeIterations.setter
    def ErodeIterations(self, val):
        self.__erode_iterations = val

    @property
    def BlurSize(self):
        return self.__blur_size

    @BlurSize.setter
    def BlurSize(self, val):
        self.__blur_size = val

    @property
    def ThresholdMin(self):
        return self.__threshold_min

    @ThresholdMin.setter
    def ThresholdMin(self, val):
        self.__threshold_min = val

    @property
    def ThresholdMax(self):
        return self.__threshold_max

    @ThresholdMax.setter
    def ThresholdMax(self, val):
        self.__threshold_max = val

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


class UnskewPageParameters:
    def __init__(self):
        self.__erode_size = (2, 2)
        self.__erode_iterations = 7

        self.__canny_min = 25
        self.__canny_max = 225
        self.__canny_aperture_size = 5

        self.__hough_lines_delta_rho = 1
        self.__hough_lines_delta_tetha = np.pi / (180 * 20)
        self.__hough_lines_threshold = 70
        self.__hough_lines_minLineLength = 300
        self.__hough_lines_maxLineGap = 90

        self.__angle_limit = 20
        self.__angle_limit_stddev = 0.5

    @property
    def ErodeSize(self):
        return self.__erode_size

    @ErodeSize.setter
    def ErodeSize(self, val):
        self.__erode_size = val

    @property
    def ErodeIterations(self):
        return self.__erode_iterations

    @ErodeIterations.setter
    def ErodeIterations(self, val):
        self.__erode_iterations = val

    @property
    def CannyMin(self):
        return self.__canny_min

    @CannyMin.setter
    def CannyMin(self, val):
        self.__canny_min = val

    @property
    def CannyMax(self):
        return self.__canny_max

    @CannyMax.setter
    def CannyMax(self, val):
        self.__canny_max = val

    @property
    def CannyApertureSize(self):
        return self.__canny_aperture_size

    @CannyApertureSize.setter
    def CannyApertureSize(self, val):
        self.__canny_aperture_size = val

    @property
    def HoughLinesDeltaRho(self):
        return self.__hough_lines_delta_rho

    @HoughLinesDeltaRho.setter
    def HoughLinesDeltaRho(self, val):
        self.__hough_lines_delta_rho = val

    @property
    def HoughLinesDeltaTetha(self):
        return self.__hough_lines_delta_tetha

    @HoughLinesDeltaTetha.setter
    def HoughLinesDeltaTetha(self, val):
        self.__hough_lines_delta_tetha = val

    @property
    def HoughLinesThreshold(self):
        return self.__hough_lines_threshold

    @HoughLinesThreshold.setter
    def HoughLinesThreshold(self, val):
        self.__hough_lines_threshold = val

    @property
    def HoughLinesMinLineLength(self):
        return self.__hough_lines_minLineLength

    @HoughLinesMinLineLength.setter
    def HoughLinesMinLineLength(self, val):
        self.__hough_lines_minLineLength = val

    @property
    def HoughLinesMaxLineGap(self):
        return self.__hough_lines_maxLineGap

    @HoughLinesMaxLineGap.setter
    def HoughLinesMaxLineGap(self, val):
        self.__hough_lines_maxLineGap = val

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
        self.__erode_size = (9, 9)
        self.__erode_iterations = 1
        self.__threshold1_min = 240
        self.__threshold1_max = 255
        self.get_rectangle_from_contour = GetRectangleFromContourParameters()
        self.__dilate_size = (2, 2)
        self.__threshold2_min = 200
        self.__threshold2_max = 255
        self.__contour_area_min = 0.002 * 0.002
        self.__contour_area_max = 0.5 * 0.5
        self.__border = 10

    @property
    def ErodeSize(self):
        return self.__erode_size

    @ErodeSize.setter
    def ErodeSize(self, val):
        self.__erode_size = val

    @property
    def ErodeIterations(self):
        return self.__erode_iterations

    @ErodeIterations.setter
    def ErodeIterations(self, val):
        self.__erode_iterations = val

    @property
    def Threshold1Min(self):
        return self.__threshold1_min

    @Threshold1Min.setter
    def Threshold1Min(self, val):
        self.__threshold1_min = val

    @property
    def Threshold1Max(self):
        return self.__threshold1_max

    @Threshold1Max.setter
    def Threshold1Max(self, val):
        self.__threshold1_max = val

    @property
    def DilateSize(self):
        return self.__dilate_size

    @DilateSize.setter
    def DilateSize(self, val):
        self.__dilate_size = val

    @property
    def Threshold2Min(self):
        return self.__threshold2_min

    @Threshold2Min.setter
    def Threshold2Min(self, val):
        self.__threshold2_min = val

    @property
    def Threshold2Max(self):
        return self.__threshold2_max

    @Threshold2Max.setter
    def Threshold2Max(self, val):
        self.__threshold2_max = val

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
            if param == "SplitTwoWavesGetRectangleFromContourMinE":
                parameters.split_two_waves.get_rectangle_from_contour.MinE = value
            elif param == "SplitTwoWavesGetRectangleFromContourMaxE":
                parameters.split_two_waves.get_rectangle_from_contour.MaxE = value
            elif param == "SplitTwoWavesGetRectangleFromContourMaxIterations":
                parameters.split_two_waves.get_rectangle_from_contour.MaxIterations = (
                    value
                )
            elif param == "SplitTwoWavesErodeSize":
                parameters.split_two_waves.ErodeSize = value
            elif param == "SplitTwoWavesErodeIterations":
                parameters.split_two_waves.ErodeIterations = value
            elif param == "SplitTwoWavesBlurSize":
                parameters.split_two_waves.BlurSize = value
            elif param == "SplitTwoWavesThresholdMin":
                parameters.split_two_waves.ThresholdMin = value
            elif param == "SplitTwoWavesThresholdMax":
                parameters.split_two_waves.ThresholdMax = value
            elif param == "SplitTwoWavesRapportRect1Rect2":
                parameters.split_two_waves.RapportRect1Rect2 = value
            elif param == "SplitTwoWavesWaveTop":
                parameters.split_two_waves.WaveTop = value
            elif param == "SplitTwoWavesWaveBottom":
                parameters.split_two_waves.WaveBottom = value
            elif param == "SplitTwoWavesWaveRight":
                parameters.split_two_waves.WaveRight = value
            elif param == "SplitTwoWavesWaveLeft":
                parameters.split_two_waves.WaveLeft = value
            elif param == "SplitTwoWavesFoundContourIterations":
                parameters.split_two_waves.FoundContourIterations = value
            elif param == "SplitTwoWavesNpoints2pages":
                parameters.split_two_waves.Npoints2pages = value
            elif param == "SplitTwoWavesNpoints1page":
                parameters.split_two_waves.Npoints1page = value
            elif param == "UnskewPageErodeSize":
                parameters.unskew_page.ErodeSize = value
            elif param == "UnskewPageErodeIterations":
                parameters.unskew_page.ErodeIterations = value
            elif param == "UnskewPageCannyMin":
                parameters.unskew_page.CannyMin = value
            elif param == "UnskewPageCannyMax":
                parameters.unskew_page.CannyMax = value
            elif param == "UnskewPageCannyApertureSize":
                parameters.unskew_page.CannyApertureSize = value
            elif param == "UnskewPageHoughLinesDeltaRho":
                parameters.unskew_page.HoughLinesDeltaRho = value
            elif param == "UnskewPageHoughLinesDeltaTetha":
                parameters.unskew_page.HoughLinesDeltaTetha = value
            elif param == "UnskewPageHoughLinesThreshold":
                parameters.unskew_page.HoughLinesThreshold = value
            elif param == "UnskewPageHoughLinesMinLineLength":
                parameters.unskew_page.HoughLinesMinLineLength = value
            elif param == "UnskewPageHoughLinesMaxLineGap":
                parameters.unskew_page.HoughLinesMaxLineGap = value
            elif param == "UnskewPageAngleLimit":
                parameters.unskew_page.AngleLimit = value
            elif param == "UnskewPageAngleLimitStddev":
                parameters.unskew_page.AngleLimitStddev = value
            elif param == "CropAroundDataInPageErodeSize":
                parameters.crop_around_data_in_page.ErodeSize = value
            elif param == "CropAroundDataInPageErodeIterations":
                parameters.crop_around_data_in_page.ErodeIterations = value
            elif param == "CropAroundDataInPageThreshold1Min":
                parameters.crop_around_data_in_page.Threshold1Min = value
            elif param == "CropAroundDataInPageThreshold1Max":
                parameters.crop_around_data_in_page.Threshold1Max = value
            elif param == "CropAroundDataInPageGetRectangleFromContourMinE":
                parameters.crop_around_data_in_page.get_rectangle_from_contour.MinE = (
                    value
                )
            elif param == "CropAroundDataInPageGetRectangleFromContourMaxE":
                parameters.crop_around_data_in_page.get_rectangle_from_contour.MaxE = (
                    value
                )
            elif param == "CropAroundDataInPageGetRectangleFromContourMaxIterations":
                parameters.crop_around_data_in_page.get_rectangle_from_contour.MaxIterations = (
                    value
                )
            elif param == "CropAroundDataInPageDilateSize":
                parameters.crop_around_data_in_page.DilateSize = value
            elif param == "CropAroundDataInPageThreshold2Min":
                parameters.crop_around_data_in_page.Threshold2Min = value
            elif param == "CropAroundDataInPageThreshold2Max":
                parameters.crop_around_data_in_page.Threshold2Max = value
            elif param == "CropAroundDataInPageContourAreaMin":
                parameters.crop_around_data_in_page.ContourAreaMin = value
            elif param == "CropAroundDataInPageContourAreaMax":
                parameters.crop_around_data_in_page.ContourAreaMax = value
            elif param == "CropAroundDataInPageBorder":
                parameters.crop_around_data_in_page.Border = value
            else:
                raise Exception("Invalid property.", param)

        return parameters

