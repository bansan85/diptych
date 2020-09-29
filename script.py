import cv2
import numpy as np
import print_release
import print_test
import sys
import parameters as par
import functools

DEBUG = True


class LOG:
    def __init__(self):
        self.t = 0


class Compute:
    def get_alpha(point1, point2):
        angle = np.arctan2(point1[1] - point2[1], point1[0] - point2[0]) / np.pi * 180
        if angle < 0:
            angle = angle + 180
        return angle

    def get_angle360(point1, point2):
        angle = np.arctan2(point1[1] - point2[1], point1[0] - point2[0]) / np.pi * 180
        return angle

    def get_alpha_posx(point1, point2):
        if point1[1] == point2[1]:
            return None, None
        angle = np.arctan2(point1[1] - point2[1], point1[0] - point2[0]) / np.pi * 180
        if angle < 0:
            angle = angle + 180
        posx = (point1[0] * point2[1] - point2[0] * point1[1]) / (point2[1] - point1[1])
        return angle, posx

    # On filtre les lignes plutôt horizontales
    def constrait_angle(data, limit_angle, objectif_ang, objectif_posx, limit_posx):
        ang, posx = data
        if ang is None or posx is None:
            return False
        angle_ok = (
            -limit_angle < ang + objectif_ang and ang + objectif_ang < limit_angle
        ) or (
            -limit_angle < ang + objectif_ang - 180
            and ang + objectif_ang - 180 < limit_angle
        )
        posx_ok = (
            objectif_posx - limit_posx <= posx and posx <= objectif_posx + limit_posx
        )
        return angle_ok and posx_ok

    def PourcentError(val1, val2):
        assert val1 >= 0
        assert val2 >= 0
        return np.absolute(val1 - val2) / np.maximum(val1, val2) * 100.0

    def IsContourRectangle(rect, tolerance):
        if len(rect) != 4:
            return False

        distance = []
        for i in range(4):
            distance.append(
                np.sqrt(
                    np.square(rect[i, 0, 1] - rect[(i + 1) % 4, 0, 1])
                    + np.square(rect[i, 0, 0] - rect[(i + 1) % 4, 0, 0])
                )
            )
        diagonale = []
        for i in range(2):
            diagonale.append(
                np.sqrt(
                    np.square(rect[i, 0, 1] - rect[i + 2, 0, 1])
                    + np.square(rect[i, 0, 0] - rect[i + 2, 0, 0])
                )
            )
        un = Compute.PourcentError(distance[0], distance[2]) < tolerance
        deux = Compute.PourcentError(distance[1], distance[3]) < tolerance
        trois = Compute.PourcentError(diagonale[0], diagonale[1]) < tolerance
        return un and deux and trois

    def line_intersection(line1, line2):
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            raise Exception("lines do not intersect")

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return int(x), int(y)


class TraitementImage:
    def charge_image(fichier):
        return cv2.imread(fichier)

    def convertion_en_niveau_de_gris(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def get_rectangle_from_contour(c, n, mine, maxe, maxiter):
        e = (mine + maxe) / 2
        edicho = (maxe - mine) / 2
        nn = 0
        cntmax = []
        lastcnt = []
        while True:
            cnt = cv2.approxPolyDP(c, e * cv2.arcLength(c, True), True)
            if len(cnt) == n:
                return cnt
            if len(cnt) > n:
                edicho = edicho / 2
                e = e + edicho
                if nn >= 0:
                    nn = nn + 1
                else:
                    nn = 1
                if len(cntmax) < n or len(cnt) < len(cntmax):
                    cntmax = cnt
            if len(cnt) < n:
                edicho = edicho / 2
                e = e - edicho
                if nn <= 0:
                    nn = nn - 1
                else:
                    nn = -1
                # On garde le contour le plus grand au cas où on ne trouve
                # pas un contour de taille suffisant.
                if len(cntmax) < len(cnt):
                    cntmax = cnt
            if np.abs(nn) > maxiter:
                return cntmax
            # Si la taille du contour change, on réinitialise le compteur.
            if len(lastcnt) != len(cnt):
                nn = 0
            if DEBUG:
                print(len(cnt), ":", e)
            lastcnt = cnt

    def rotate_image(image, angle_deg):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle_deg, 1.0)
        result = cv2.warpAffine(
            image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR
        )
        return result

    def crop_rectangle(image, min_x, max_x, min_y, max_y):
        return image[min_y:max_y, min_x:max_x]

    def is_black_white(image):
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        return sum(hist[1:255]) < 1


class SeparatePage:
    def FindImages(imagee, erode_pourcent_size):
        INTERNAL_BORDER = 20
        blurimg = cv2.blur(cv2.cvtColor(imagee, cv2.COLOR_BGR2GRAY), (10, 10))
        white = [255]
        blurimg_ = cv2.copyMakeBorder(
            blurimg,
            INTERNAL_BORDER,
            INTERNAL_BORDER,
            INTERNAL_BORDER,
            INTERNAL_BORDER,
            cv2.BORDER_CONSTANT,
            value=white,
        )
        cv2.imwrite("A_1.png", blurimg_)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        cv2.imwrite("A_2.png", blurimg_)
        blurimg3 = cv2.dilate(blurimg_, kernel)
        cv2.imwrite("A_3.png", blurimg3)
        _, blurimg4 = cv2.threshold(blurimg3, 254, 255, cv2.THRESH_TOZERO)
        cv2.imwrite("A_4.png", blurimg4)
        _, blurimg5 = cv2.threshold(blurimg4, 0, 255, cv2.THRESH_BINARY_INV)
        cv2.imwrite("A_5.png", blurimg5)

        blurimg6 = cv2.morphologyEx(blurimg5, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite("A_6.png", blurimg6)
        blurimg7 = cv2.morphologyEx(blurimg6, cv2.MORPH_OPEN, kernel)
        cv2.imwrite("A_7.png", blurimg7)

        contours, _ = cv2.findContours(
            blurimg7, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in contours:
            for contour in cnt:
                x, y = contour[0]
                contour[0, 0] = x - INTERNAL_BORDER
                contour[0, 1] = y - INTERNAL_BORDER
        img_tmp = cv2.drawContours(np.copy(imagee), contours, -1, (0, 0, 255), 3)
        cv2.imwrite("A_8.png", img_tmp)
        img_mask = np.zeros(blurimg.shape, np.uint8)
        img_mask_erode = np.zeros(blurimg.shape, np.uint8)
        himg, wimg, _ = imagee.shape
        for cnt in contours:
            c = TraitementImage.get_rectangle_from_contour(
                cnt,
                4,
                0.0000001,
                0.9999,
                10,
            )
            if cv2.contourArea(c) > 0.01 * himg * wimg:
                img_tmp = cv2.drawContours(img_tmp, [c], -1, (255, 0, 0), 3)
                img_mask = cv2.drawContours(img_mask, [c], -1, (255, 0, 0), -1)
                img_mask_erodei = np.zeros(blurimg.shape, np.uint8)
                img_mask_erodei = cv2.drawContours(
                    img_mask_erodei, [c], -1, (255, 0, 0), -1
                )
                x, y, w, h = cv2.boundingRect(c)
                erode_pixel = int(np.minimum(w, h) * erode_pourcent_size)
                img_mask_erodei = cv2.erode(
                    img_mask_erodei,
                    np.ones((3, 3), np.uint8),
                    iterations=erode_pixel,
                )
                img_mask_erode = cv2.bitwise_or(img_mask_erode, img_mask_erodei)

        cv2.imwrite("A_9.png", img_tmp)
        cv2.imwrite("A_10.png", img_mask)
        cv2.imwrite("A_11.png", img_mask_erode)
        return img_mask_erode

    def found_split_line_with_line(image, parameters):
        if DEBUG:
            cv2.imwrite("1_1.png", image)
        gray = TraitementImage.convertion_en_niveau_de_gris(image)
        if TraitementImage.is_black_white(gray):
            blurimg = cv2.blur(gray, parameters.BlurSize)
        else:
            blurimg = gray
        if DEBUG:
            cv2.imwrite("1_2.png", blurimg)

        for thresholdi in parameters.ThresholdMin:
            # On repasse l'image en noir et blanc.
            _, threshold = cv2.threshold(
                blurimg,
                thresholdi,
                255,
                cv2.THRESH_BINARY,
            )
            if DEBUG:
                cv2.imwrite("1_3_" + str(thresholdi) + ".png", threshold)
            eroded = cv2.erode(
                threshold,
                np.ones(parameters.Erode.Size),
                iterations=parameters.Erode.Iterations,
            )
            if DEBUG:
                cv2.imwrite("1_4_" + str(thresholdi) + ".png", eroded)
            canny = cv2.Canny(
                eroded,
                parameters.Canny.Min,
                parameters.Canny.Max,
                apertureSize=parameters.Canny.ApertureSize,
            )
            if DEBUG:
                cv2.imwrite("1_5_" + str(thresholdi) + ".png", canny)
            list_lines = cv2.HoughLinesP(
                canny,
                parameters.HoughLines.DeltaRho,
                parameters.HoughLines.DeltaTetha,
                parameters.HoughLines.Threshold,
                minLineLength=parameters.HoughLines.MinLineLength,
                maxLineGap=parameters.HoughLines.MaxLineGap,
            )
            lines = list(map(lambda list_line: list_line[0], list_lines))
            lines_angle_posx = list(
                map(lambda p: Compute.get_alpha_posx((p[0], p[1]), (p[2], p[3])), lines)
            )
            if DEBUG:
                image_with_lines = image.copy()
                for a, b, c, d in lines:
                    cv2.line(
                        image_with_lines,
                        (a, b),
                        (c, d),
                        (0, 0, 255),
                        1,
                    )
                cv2.imwrite("1_6_" + str(thresholdi) + ".png", image_with_lines)

            h, w, _ = image.shape

            valid_lines = list(
                filter(
                    lambda x: Compute.constrait_angle(
                        x, parameters.Limit.Tetha, 90, w / 2, parameters.Limit.Rho
                    ),
                    lines_angle_posx,
                )
            )

            if len(valid_lines) != 0:
                break

        if len(valid_lines) == 0:
            raise Exception(
                "found_split_line_with_line",
                "Failed to find candidate for the separator line.",
            )

        list_of_angles = list(map(lambda x: x[0], valid_lines))
        list_of_posx = list(map(lambda x: x[1], valid_lines))
        ecarttype_angles = np.std(list_of_angles)
        moyenne_angles = np.mean(list_of_angles)
        ecarttype_posx = np.std(list_of_posx)
        moyenne_posx = np.mean(list_of_posx)
        angle_dans_ecarttype = list(
            filter(
                lambda x: moyenne_angles - ecarttype_angles <= x[0]
                and x[0] <= moyenne_angles + ecarttype_angles
                and moyenne_posx - ecarttype_posx <= x[1]
                and x[1] <= moyenne_posx + ecarttype_posx,
                valid_lines,
            )
        )

        def sum_tmp(x1, x2):
            return x1[0] + x2[0], x1[1] + x2[1]

        angle, posx = functools.reduce(sum_tmp, angle_dans_ecarttype)
        angle_1 = angle / len(angle_dans_ecarttype)
        posx_1 = posx / len(angle_dans_ecarttype)
        point_1a = (int(posx_1), 0)
        point_1b = (
            int(posx_1 - np.tan((angle_1 - 90.0) / 180.0 * np.pi) * h),
            h - 1,
        )

        if DEBUG:
            image_with_lines = image.copy()
            for angle_i, posx_i in angle_dans_ecarttype:
                point_ia = (int(posx_i), 0)
                point_ib = (
                    int(posx_i - np.tan((angle_i - 90.0) / 180.0 * np.pi) * h),
                    h - 1,
                )
                cv2.line(
                    image_with_lines,
                    (point_ia[0], point_ia[1]),
                    (point_ib[0], point_ib[1]),
                    (0, 0, 255),
                    1,
                )
            cv2.line(
                image_with_lines,
                (point_1a[0], point_1a[1]),
                (point_1b[0], point_1b[1]),
                (255, 0, 0),
                5,
            )
            cv2.imwrite("1_7.png", image_with_lines)

        return angle_1, posx_1

    def found_split_line_with_wave(image, parameters):
        if DEBUG:
            cv2.imwrite("2_1.png", image)
        gray = TraitementImage.convertion_en_niveau_de_gris(image)
        blurimg = gray
        if TraitementImage.is_black_white(gray):
            blurimg = cv2.blur(gray, (5, 5))
        if DEBUG:
            cv2.imwrite("2_2.png", blurimg)
        # On repasse l'image en noir et blanc.
        _, threshold = cv2.threshold(
            blurimg,
            parameters.ThresholdMin[0],
            255,
            cv2.THRESH_BINARY,
        )
        if DEBUG:
            cv2.imwrite("2_3.png", threshold)
        eroded = cv2.erode(
            threshold,
            np.ones(parameters.Erode.Size),
            iterations=parameters.Erode.Iterations,
        )
        if DEBUG:
            cv2.imwrite("2_4.png", eroded)
        white = [0]
        SIZE_BORDER = 20
        ee = cv2.copyMakeBorder(
            eroded,
            SIZE_BORDER,
            SIZE_BORDER,
            SIZE_BORDER,
            SIZE_BORDER,
            cv2.BORDER_CONSTANT,
            value=white,
        )
        if DEBUG:
            cv2.imwrite("2_4a.png", ee)

        # On cherche tous les contours
        contours, _ = cv2.findContours(ee, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # pour ne garder que le plus grand. Normalement, cela doit être celui qui fait le contour des pages
        # et non le tour du bord de l'image.
        # On tri les contours du plus grand au plus petit.
        # L'intérêt est de comparer la taille du premier et du deuxième contour.
        # Si le bord de la page est trop en haut et en bas, plutôt que d'avoir
        # un contour qui fait les deux pages, on peut avoir deux contours qui
        # font chaque page.
        cs = sorted(contours, key=lambda x: 1.0 / (np.maximum(cv2.contourArea(x), 1)))
        if DEBUG:
            img_tmp = cv2.cvtColor(ee, cv2.COLOR_GRAY2BGR)
            img_tmp = cv2.drawContours(img_tmp, cs, 0, (255, 0, 0), 10)
            img_tmp = cv2.drawContours(img_tmp, cs, 1, (0, 255, 0), 10)
            cv2.imwrite("2_4b.png", img_tmp)

        h, w, _ = image.shape
        for i in range(len(cs)):
            for j in range(len(cs[i])):
                x, y = cs[i][j, 0]
                cs[i][j, 0, 0] = np.minimum(x - SIZE_BORDER, w - 1)
                cs[i][j, 0, 1] = np.minimum(y - SIZE_BORDER, h - 1)
        xx = SeparatePage.FindImages(image, parameters.ReduceImageWhenDetecting)

        def filtremask(x):
            return xx[x[0, 1], x[0, 0]] == 255

        cs2 = []
        for i in range(len(cs)):
            csi = []
            for j in range(len(cs[i])):
                if xx[cs[i][j, 0, 1], cs[i][j, 0, 0]] == 0:
                    csi.append(cs[i][j, 0])
            csibis = np.ndarray((len(csi), 1, 2), dtype=np.int32)
            for j in range(len(csi)):
                csibis[j, 0, 0] = csi[j][0]
                csibis[j, 0, 1] = csi[j][1]
            cs2.append(csibis)
        size_cs1 = cv2.contourArea(cs2[0])
        size_cs2 = cv2.contourArea(cs2[1])

        nb_rectangle = int(size_cs1 / size_cs2 < parameters.RapportRect1Rect2) + 1
        if DEBUG:
            img_tmp = cv2.cvtColor(eroded, cv2.COLOR_GRAY2BGR)
            for i in range(nb_rectangle):
                img_tmp = cv2.drawContours(img_tmp, cs2, i, (255, 0, 0), 10)
            img_tmp = cv2.drawContours(img_tmp, cs2, nb_rectangle, (0, 0, 255), 3)
            cv2.imwrite("2_5.png", img_tmp)

        # cnt = cv2.approxPolyDP(c, 0.001*cv2.arcLength(c,True),True)
        # Il faut au minimum 10 points pour détecter le décroché qui indique la séparation entre deux pages.
        if nb_rectangle == 1:
            cslist = [cs2[0]]
            # Si on a un seul contour pour la double page, la vague peut être caractérisée par 3 points.
            nb_point_in_wave = 3
        else:
            cslist = [cs2[0], cs2[1]]
            # Si on a deux contours pour la double page, la vague ne peut être caractérisée que par 2 points.
            nb_point_in_wave = 2
        toppoints = []
        bottompoints = []
        for c in cslist:
            if nb_rectangle == 1:
                n = parameters.Npoints2pages
            else:
                n = parameters.Npoints1page
            nloop = 0
            while nloop < parameters.FoundContourIterations:
                cnt = TraitementImage.get_rectangle_from_contour(
                    c,
                    n,
                    parameters.RectangleContour.MinE,
                    parameters.RectangleContour.MaxE,
                    parameters.RectangleContour.MaxIterations,
                )
                if DEBUG:
                    test = cv2.drawContours(np.copy(image), [cnt], 0, (0, 0, 255), 10)
                    for i in range(nb_rectangle):
                        cv2.drawContours(test, cs2, i, (255, 0, 0), 10)
                    cv2.imwrite("2_6_" + str(nloop) + ".png", test)
                h, w, _ = image.shape
                toppointsi = []
                bottompointsi = []
                # On filtre les points qui sont à peu près au centre (x/2) de l'image
                for ci in cnt:
                    x, y = ci[0]
                    if parameters.WaveLeft * w < x and x < parameters.WaveRight * w:
                        if y < parameters.WaveTop * h:
                            toppointsi.append((x, y))
                        elif y > parameters.WaveBottom * h:
                            bottompointsi.append((x, y))
                # Est-ce qu'on a suffisamment de points ?
                if (
                    len(toppointsi) >= nb_point_in_wave
                    and len(bottompointsi) >= nb_point_in_wave
                ):
                    toppoints.append(toppointsi)
                    bottompoints.append(bottompointsi)
                    break
                n = len(cnt) + 1
                nloop = nloop + 1

            if nloop == parameters.FoundContourIterations:
                raise Exception(
                    "Failed to found contour of the page.",
                    str(parameters.FoundContourIterations) + " iterations",
                )

        # Le point de séparation des deux pages en haut et bas
        maxtoppoints = list(map(lambda x: max(x, key=lambda x: x[1]), toppoints))
        minbottompoints = list(map(lambda x: min(x, key=lambda x: x[1]), bottompoints))
        toppoint = (
            int(sum(x for x, y in maxtoppoints) / len(maxtoppoints)),
            int(sum(y for x, y in maxtoppoints) / len(maxtoppoints)),
        )
        bottompoint = (
            int(sum(x for x, y in minbottompoints) / len(minbottompoints)),
            int(sum(y for x, y in minbottompoints) / len(minbottompoints)),
        )
        if DEBUG:
            image_with_lines = image.copy()
            cv2.line(
                image_with_lines,
                toppoint,
                bottompoint,
                (0, 0, 255),
                1,
            )
            cv2.imwrite("2_7.png", image_with_lines)
        return Compute.get_alpha_posx(bottompoint, toppoint)

    # Sépare une page en deux en détectant la vague dans le papier en haut et en bas de la reliure.
    def split_two_waves(
        image,
        parameters,
    ):
        param1 = par.FoundSplitLineWithLineParameters(
            parameters.BlurSize,
            parameters.ThresholdMin,
            parameters.Erode,
            parameters.Canny,
            parameters.HoughLines,
            parameters.DeltaRhoTetha,
        )
        angle1, pos1 = SeparatePage.found_split_line_with_line(image, param1)

        param2 = par.FoundSplitLineWithWave(
            parameters.BlurSize,
            parameters.ThresholdMin,
            parameters.Erode,
            parameters.ReduceImageWhenDetecting,
            parameters.RapportRect1Rect2,
            parameters.Npoints2pages,
            parameters.Npoints1page,
            parameters.FoundContourIterations,
            parameters.RectangleContour,
            parameters.WaveTop,
            parameters.WaveBottom,
            parameters.WaveLeft,
            parameters.WaveRight,
        )
        angle2, pos2 = SeparatePage.found_split_line_with_wave(image, param2)

        pos_moy = (pos1 + pos2) / 2
        angle_moy = (angle1 + angle2) / 2

        h, w, _ = image.shape

        toppoint = (int(pos_moy), 0)
        bottompoint = (
            int(pos_moy - np.tan((angle_moy - 90.0) / 180.0 * np.pi) * h),
            h - 1,
        )

        LOG.OUTPUT.print("separation double page angle", (angle1 + angle2) / 2)
        LOG.OUTPUT.print("separation double page y=0", (pos1 + pos2) / 2)

        # On défini le masque pour séparer la page droite et gauche
        mask = np.zeros((h, w), np.uint8)
        pts = np.array(
            [
                [0, 0],
                [toppoint[0], 0],
                [toppoint[0], toppoint[1]],
                [bottompoint[0], bottompoint[1]],
                [bottompoint[0], h - 1],
                [0, h - 1],
            ]
        )
        mask2 = cv2.drawContours(mask, np.int32([pts]), 0, 255, -1)
        page_gauche = image.copy()
        page_droite = image.copy()
        # On applique le masque
        page_gauche[mask2 == 0] = 0
        page_droite[mask2 > 0] = 0
        # On crop les images.
        page_gauche_0 = page_gauche[:, 0 : np.maximum(toppoint[0], bottompoint[0]) - 1]
        page_droite_0 = page_droite[:, np.minimum(toppoint[0], bottompoint[0]) :]
        if DEBUG:
            cv2.imwrite("3_1.png", page_gauche_0)
            cv2.imwrite("3_2.png", page_droite_0)

        # On renvoie les images cropées.
        return page_gauche_0, page_droite_0

    def unskew_page(image, n, parameters):
        if DEBUG:
            cv2.imwrite("4_" + str(n) + "_1.png", image)
        img_gauche = TraitementImage.convertion_en_niveau_de_gris(image)
        # On grossit les images pour former un boudin et mieux détecter les lignes.
        eroded = cv2.erode(
            img_gauche,
            np.ones(parameters.Erode.Size),
            iterations=parameters.Erode.Iterations,
        )
        if DEBUG:
            cv2.imwrite("4_" + str(n) + "_2.png", eroded)

        # Aide à la détection des contours
        canny = cv2.Canny(
            eroded,
            parameters.Canny.Min,
            parameters.Canny.Max,
            apertureSize=parameters.Canny.ApertureSize,
        )
        if DEBUG:
            cv2.imwrite("4_" + str(n) + "_3.png", canny)

        # Détection des lignes.
        # La précision doit être de l'ordre de 0.05°
        list_lines = cv2.HoughLinesP(
            canny,
            parameters.HoughLines.DeltaRho,
            parameters.HoughLines.DeltaTetha,
            parameters.HoughLines.Threshold,
            minLineLength=parameters.HoughLines.MinLineLength,
            maxLineGap=parameters.HoughLines.MaxLineGap,
        )

        # lines contient une liste de liste de lignes.
        # Le deuxième niveau de liste ne contient toujours qu'une ligne.
        lines = list(map(lambda list_line: list_line[0], list_lines))

        # On filtre les lignes plutôt horizontales / verticales
        def is_angle_vertical_or_horizontal(line, limit_angle=parameters.AngleLimit):
            a, b, c, d = line
            angl = (np.arctan2(d - b, c - a) / np.pi * 180 + 180) % 90
            return angl < limit_angle or angl > 90 - limit_angle

        valid_lines = list(filter(is_angle_vertical_or_horizontal, lines))

        if DEBUG:
            image_with_lines = image.copy()
            for line_x1, line_y1, line_x2, line_y2 in lines:
                cv2.line(
                    image_with_lines,
                    (line_x1, line_y1),
                    (line_x2, line_y2),
                    (255, 0, 0),
                    1,
                )
            for line_x1, line_y1, line_x2, line_y2 in valid_lines:
                cv2.line(
                    image_with_lines,
                    (line_x1, line_y1),
                    (line_x2, line_y2),
                    (0, 0, 255),
                    1,
                )
            cv2.imwrite("4_" + str(n) + "_4.png", image_with_lines)

        # On converti les lignes en angles
        def convert_line_to_angle_closed_to_zero(line, limit_angle):
            angle = (
                np.arctan2(line[3] - line[1], line[2] - line[0]) / np.pi * 180 + 180
            ) % 180
            if angle > 180 - limit_angle:
                return angle - 180
            if angle > 90 - limit_angle:
                return angle - 90
            return angle

        angles = list(
            map(
                lambda x: convert_line_to_angle_closed_to_zero(
                    x, parameters.AngleLimit
                ),
                valid_lines,
            )
        )

        # On enlève les valeurs extrêmes
        ecarttype = np.std(angles) * parameters.AngleLimitStddev
        moyenne = np.mean(angles)
        angle_dans_ecarttype = list(
            filter(
                lambda x: moyenne - ecarttype < x and x < moyenne + ecarttype, angles
            )
        )

        if len(angle_dans_ecarttype) == 0:
            raise Exception(
                "Image " + str(n) + ": no angle have the defined constraints.",
                "AngleLimitStddev too small ?",
            )
        rotate_angle = np.mean(angle_dans_ecarttype)
        LOG.OUTPUT.print("page rotation " + str(n), rotate_angle)
        # Enfin, on tourne.
        img_rotated = TraitementImage.rotate_image(image, rotate_angle)
        if DEBUG:
            cv2.imwrite("4_" + str(n) + "_5.png", img_rotated)
        return img_rotated

    def crop_around_data_in_page(image, n, parameters):
        # On force le noir et blanc
        if DEBUG:
            cv2.imwrite("5_" + str(n) + "_1.png", image)
        gray = TraitementImage.convertion_en_niveau_de_gris(image)
        eroded = cv2.erode(
            gray, np.ones(parameters.Erode.Size), iterations=parameters.Erode.Iterations
        )
        if DEBUG:
            cv2.imwrite("5_" + str(n) + "_2.png", eroded)
        _, threshold = cv2.threshold(
            eroded,
            parameters.Threshold1.Min,
            parameters.Threshold1.Max,
            cv2.THRESH_BINARY,
        )
        if DEBUG:
            cv2.imwrite("5_" + str(n) + "_3.png", threshold)
        # On récupère le contour le plus grand.
        contours, _ = cv2.findContours(
            threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        c = max(contours, key=cv2.contourArea)
        image2222 = cv2.drawContours(image.copy(), contours, -1, (0, 0, 255), 3)
        if DEBUG:
            cv2.imwrite("5_" + str(n) + "_4.png", image2222)

        # On garde le rectangle le plus grand.
        rect = TraitementImage.get_rectangle_from_contour(
            c,
            4,
            parameters.RectangleContour.MinE,
            parameters.RectangleContour.MaxE,
            parameters.RectangleContour.MaxIterations,
        )
        image22223 = cv2.drawContours(image2222, [rect], -1, (255, 0, 0), 3)
        if DEBUG:
            cv2.imwrite("5_" + str(n) + "_5.png", image22223)
        # Si on n'a pas de rectangle, on essaie de trouver le contour de la page avec les traits horizontaux et verticaux.
        if not Compute.IsContourRectangle(rect, parameters.PourcentageEcartRectangle):
            blurimg = cv2.blur(gray, (10, 10))
            if DEBUG:
                cv2.imwrite("5_" + str(n) + "_5a.png", blurimg)
            _, threshold = cv2.threshold(
                blurimg,
                200,  # Plus c'est petit, plus c'est clair.
                255,
                cv2.THRESH_BINARY,
            )
            if DEBUG:
                cv2.imwrite("5_" + str(n) + "_5b.png", threshold)
            canny = cv2.Canny(
                threshold,
                25,
                255,
                apertureSize=5,
            )
            if DEBUG:
                cv2.imwrite("5_" + str(n) + "_5c.png", canny)
            lines1 = cv2.HoughLinesP(
                canny,
                1,
                np.pi / (180 * 20),
                30,
                minLineLength=100,
                maxLineGap=30,
            )
            if DEBUG:
                image_with_lines = image.copy()
                for line in lines1:
                    for a, b, c, d in line:
                        cv2.line(
                            image_with_lines,
                            (a, b),
                            (c, d),
                            (0, 0, 255),
                            1,
                        )
                cv2.imwrite("5_" + str(n) + "_5d.png", image_with_lines)

            blurimg__ = cv2.equalizeHist(blurimg)
            if DEBUG:
                cv2.imwrite("5_" + str(n) + "_5e.png", blurimg__)
            _, threshold = cv2.threshold(
                blurimg__,
                15,  # Plus c'est petit, plus c'est clair.
                255,
                cv2.THRESH_BINARY,
            )
            if DEBUG:
                cv2.imwrite("5_" + str(n) + "_5f.png", threshold)
            canny = cv2.Canny(
                threshold,
                25,
                255,
                apertureSize=5,
            )
            if DEBUG:
                cv2.imwrite("5_" + str(n) + "_5g.png", canny)

            # Détection des lignes.
            # La précision doit être de l'ordre de 0.05°
            lines2 = cv2.HoughLinesP(
                canny,
                1,
                np.pi / (180 * 20),
                30,
                minLineLength=100,
                maxLineGap=30,
            )
            if DEBUG:
                image_with_lines = image.copy()
                for line in lines2:
                    for a, b, c, d in line:
                        cv2.line(
                            image_with_lines,
                            (a, b),
                            (c, d),
                            (0, 0, 255),
                            1,
                        )
                cv2.imwrite("5_" + str(n) + "_5h.png", image_with_lines)

            lines_90_angle = []
            lines_0_angle = []
            DELTA_ANGLE = 3
            for liste in (lines1, lines2):
                for line in liste:
                    for a, b, c, d in line:
                        angle = Compute.get_alpha((a, b), (c, d))
                        if 90 - DELTA_ANGLE <= angle and angle <= 90 + DELTA_ANGLE:
                            lines_90_angle.append((a, b, c, d))
                        if angle <= DELTA_ANGLE or angle > 180 - DELTA_ANGLE:
                            lines_0_angle.append((a, b, c, d))

            xx = SeparatePage.FindImages(image, 0.02)
            area = 0
            cnt = []

            for i in range(len(lines_90_angle)):
                for j in range(i + 1, len(lines_90_angle)):
                    for k in range(len(lines_0_angle)):
                        for l in range(k + 1, len(lines_0_angle)):
                            x1, y1 = Compute.line_intersection(
                                (
                                    (lines_90_angle[i][0], lines_90_angle[i][1]),
                                    (lines_90_angle[i][2], lines_90_angle[i][3]),
                                ),
                                (
                                    (lines_0_angle[k][0], lines_0_angle[k][1]),
                                    (lines_0_angle[k][2], lines_0_angle[k][3]),
                                ),
                            )
                            x2, y2 = Compute.line_intersection(
                                (
                                    (lines_90_angle[i][0], lines_90_angle[i][1]),
                                    (lines_90_angle[i][2], lines_90_angle[i][3]),
                                ),
                                (
                                    (lines_0_angle[l][0], lines_0_angle[l][1]),
                                    (lines_0_angle[l][2], lines_0_angle[l][3]),
                                ),
                            )
                            x3, y3 = Compute.line_intersection(
                                (
                                    (lines_90_angle[j][0], lines_90_angle[j][1]),
                                    (lines_90_angle[j][2], lines_90_angle[j][3]),
                                ),
                                (
                                    (lines_0_angle[k][0], lines_0_angle[k][1]),
                                    (lines_0_angle[k][2], lines_0_angle[k][3]),
                                ),
                            )
                            x4, y4 = Compute.line_intersection(
                                (
                                    (lines_90_angle[j][0], lines_90_angle[j][1]),
                                    (lines_90_angle[j][2], lines_90_angle[j][3]),
                                ),
                                (
                                    (lines_0_angle[l][0], lines_0_angle[l][1]),
                                    (lines_0_angle[l][2], lines_0_angle[l][3]),
                                ),
                            )
                            xmoy = (x1 + x2 + x3 + x4) / 4
                            ymoy = (y1 + y2 + y3 + y4) / 4
                            list_of_points = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

                            def get_angle(x):
                                return Compute.get_angle360((xmoy, ymoy), (x[0], x[1]))

                            list_of_points.sort(key=get_angle)
                            cnti = np.asarray(list_of_points)
                            areai = cv2.contourArea(cnti)
                            mask = np.zeros(xx.shape, np.uint8)
                            mask = cv2.drawContours(
                                mask, [cnti], -1, (255, 255, 255), -1
                            )
                            mask = cv2.bitwise_and(xx, mask)

                            difference = cv2.subtract(xx, mask)
                            if cv2.countNonZero(difference) == 0:
                                if area == 0 or area > areai:
                                    area = areai
                                    cnt = cnti

            if area == 0:
                raise Exception("Contour not rectangular", rect)

            rect = TraitementImage.get_rectangle_from_contour(
                cnt,
                4,
                parameters.RectangleContour.MinE,
                parameters.RectangleContour.MaxE,
                parameters.RectangleContour.MaxIterations,
            )

        x_crop1 = [rect[0, 0, 0], rect[1, 0, 0], rect[2, 0, 0], rect[3, 0, 0]]
        y_crop1 = [rect[0, 0, 1], rect[1, 0, 1], rect[2, 0, 1], rect[3, 0, 1]]
        x_crop1.sort()
        y_crop1.sort()

        page_gauche_0 = image[y_crop1[1] : y_crop1[2], x_crop1[1] : x_crop1[2]]
        if DEBUG:
            cv2.imwrite("5_" + str(n) + "_6.png", page_gauche_0)

        # On enlève les bordures noirs sur le bord des pages.
        imgh, imgw, _ = page_gauche_0.shape
        min_x, min_y = imgw, imgh
        max_x = max_y = 0

        gray = TraitementImage.convertion_en_niveau_de_gris(page_gauche_0)
        dilated = cv2.dilate(gray, np.ones(parameters.DilateSize))
        if DEBUG:
            cv2.imwrite("5_" + str(n) + "_7.png", dilated)

        _, threshold = cv2.threshold(
            dilated,
            parameters.Threshold2.Min,
            parameters.Threshold2.Max,
            cv2.THRESH_BINARY,
        )
        if DEBUG:
            cv2.imwrite("5_" + str(n) + "_8.png", dilated)
        contours, _ = cv2.findContours(
            threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        image2222 = page_gauche_0.copy()
        ncontour_good_size = 0
        for _, cnt in enumerate(contours):
            if (
                parameters.ContourAreaMin * imgh * imgw < cv2.contourArea(cnt)
                and cv2.contourArea(cnt) < parameters.ContourAreaMax * imgh * imgw
            ):
                (x, y, w, h) = cv2.boundingRect(cnt)
                cv2.drawContours(image2222, [cnt], -1, (0, 0, 255), 3)
                min_x, max_x = min(x, min_x), max(x + w, max_x)
                min_y, max_y = min(y, min_y), max(y + h, max_y)
                ncontour_good_size = ncontour_good_size + 1
        if DEBUG:
            cv2.imwrite("5_" + str(n) + "_9.png", image2222)

        # Aucun contour, on renvoie une image
        if ncontour_good_size == 0:
            LOG.OUTPUT.print("image " + str(n) + " crop x1", int(imgw / 2) - 1)
            LOG.OUTPUT.print("image " + str(n) + " crop y1", int(imgh / 2) - 1)
            LOG.OUTPUT.print("image " + str(n) + " crop x2", int(imgw / 2))
            LOG.OUTPUT.print("image " + str(n) + " crop y2", int(imgh / 2))

            return (
                page_gauche_0,
                int(imgw / 2) - 1,
                int(imgw / 2),
                int(imgh / 2) - 1,
                int(imgh / 2),
                imgw,
                imgh,
            )

        LOG.OUTPUT.print("image " + str(n) + " crop x1", x_crop1[1] + min_x)
        LOG.OUTPUT.print("image " + str(n) + " crop y1", y_crop1[1] + min_y)
        LOG.OUTPUT.print("image " + str(n) + " crop x2", x_crop1[1] + max_x)
        LOG.OUTPUT.print("image " + str(n) + " crop y2", y_crop1[1] + max_y)

        return (
            page_gauche_0,
            np.maximum(min_x - parameters.Border, 0),
            np.minimum(max_x + parameters.Border, imgw - 1),
            np.maximum(min_y - parameters.Border, 0),
            np.minimum(max_y + parameters.Border, imgh - 1),
            imgw,
            imgh,
        )

    def find_dpi(self, imgw, imgh, width_paper, height_paper):
        if imgw / 200 * 2.54 < width_paper and imgh / 200 * 2.54 < height_paper:
            return 200
        elif imgw / 300 * 2.54 < width_paper and imgh / 300 * 2.54 < height_paper:
            return 300
        else:
            raise Exception("dpi", "non détecté")

    def recadre(
        self,
        image,
        n,
        width_paper,
        height_paper,
        min_x,
        max_x,
        min_y,
        max_y,
        imgw,
        imgh,
    ):
        h, w, _ = image.shape

        dpi = self.find_dpi(imgw, imgh, width_paper, height_paper)
        LOG.OUTPUT.print("image " + str(n) + " dpi", dpi)

        marge_haute_px = min_y
        marge_basse_px = imgh - max_y
        marge_gauche_px = min_x
        marge_droite_px = imgw - max_x
        white = [255, 255, 255]
        if marge_gauche_px + w + marge_droite_px < width_paper / 2.54 * dpi:
            pixels_manquant = width_paper / 2.54 * dpi - w
            image_recadre = cv2.copyMakeBorder(
                image,
                0,
                0,
                int(pixels_manquant / 2.0),
                int(pixels_manquant / 2.0),
                cv2.BORDER_CONSTANT,
                value=white,
            )
        else:
            raise Exception("marge", "marge_gauche_px")
        if marge_haute_px + h + marge_basse_px < height_paper / 2.54 * dpi:
            pixels_manquant = height_paper / 2.54 * dpi - h
            pourcenthaut = marge_haute_px / (marge_haute_px + marge_basse_px)
            pourcentbas = marge_basse_px / (marge_haute_px + marge_basse_px)
            image_recadre2 = cv2.copyMakeBorder(
                image_recadre,
                int(pixels_manquant * pourcenthaut),
                int(pixels_manquant * pourcentbas),
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=white,
            )
        else:
            raise Exception("marge", "marge_gauche_px")
        return image_recadre2

    def treat_file(self, filename, dict_test=None, dict_default_values=dict()):
        print(filename)
        img = TraitementImage.charge_image(filename)
        if img is None:
            raise Exception("Failed to load image.", filename)

        if dict_test is None:
            LOG.OUTPUT = print_release.PrintRelease()
        else:
            LOG.OUTPUT = print_test.PrintTest(dict_test)

        self.parameters = par.Parameters.init_default_values(dict_default_values)

        image1, image2 = SeparatePage.split_two_waves(
            img, self.parameters.split_two_waves
        )

        image1a = SeparatePage.unskew_page(image1, 1, self.parameters.unskew_page)
        image2a = SeparatePage.unskew_page(image2, 2, self.parameters.unskew_page)

        (
            image1a2,
            min_x1,
            max_x1,
            min_y1,
            max_y1,
            imgw1,
            imgh1,
        ) = SeparatePage.crop_around_data_in_page(
            image1a, 1, self.parameters.crop_around_data_in_page
        )
        image1b = TraitementImage.crop_rectangle(
            image1a2, min_x1, max_x1, min_y1, max_y1
        )
        (
            image2a2,
            min_x2,
            max_x2,
            min_y2,
            max_y2,
            imgw2,
            imgh2,
        ) = SeparatePage.crop_around_data_in_page(
            image2a, 2, self.parameters.crop_around_data_in_page
        )
        image2b = TraitementImage.crop_rectangle(
            image2a2, min_x2, max_x2, min_y2, max_y2
        )

        if DEBUG:
            cv2.imwrite("5_1_10.png", image1b)
            cv2.imwrite("5_2_10.png", image2b)

        image1c = self.recadre(
            image1b, 1, 21.0, 29.7, min_x1, max_x1, min_y1, max_y1, imgw1, imgh1
        )
        image2c = self.recadre(
            image2b, 2, 21.0, 29.7, min_x2, max_x2, min_y2, max_y2, imgw2, imgh2
        )

        cv2.imwrite("6_1.png", image1c)
        cv2.imwrite("6_2.png", image2c)
