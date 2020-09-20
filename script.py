import cv2
import numpy as np
import print_release
import print_test

import sys

DEBUG = True


class SeparatePage:
    def charge_image(self, fichier):
        return cv2.imread(fichier)

    def convertion_en_niveau_de_gris(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def get_rectangle_from_contour(self, c, n):
        mine = 0.00001
        maxe = 0.99
        e = (mine + maxe) / 2
        edicho = e
        nn = 0
        cntmax = []
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
                if len(cntmax) == 0 or len(cnt) < len(cntmax):
                    cntmax = cnt
            if len(cnt) < n:
                edicho = edicho / 2
                e = e - edicho
                if nn <= 0:
                    nn = nn - 1
                else:
                    nn = -1
            if np.abs(nn) > 10:
                return cntmax

    def separe_double_page_en_deux(self, image):
        gray = self.convertion_en_niveau_de_gris(image)
        eroded = cv2.erode(gray, np.ones((4, 4)), iterations=1)
        if DEBUG:
            cv2.imwrite("0_1.png", eroded)
        # Pour l'instant, l'image de base est déjà en noir et blanc.
        _, threshold = cv2.threshold(eroded, 125, 255, cv2.THRESH_BINARY)
        if DEBUG:
            cv2.imwrite("0_2.png", threshold)
        # On cherche tous les contours
        contours, _ = cv2.findContours(
            threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        # pour ne garder que le plus grand. Normalement, cela doit être celui qui fait le contour des pages
        # et non le tour du bord de l'image.
        c = max(contours, key=cv2.contourArea)
        if DEBUG:
            test = cv2.drawContours(np.copy(image), [c], 0, (0, 0, 255), 10)
            cv2.imwrite("0_3.png", test)

        # cnt = cv2.approxPolyDP(c, 0.001*cv2.arcLength(c,True),True)
        # Il faut au minimum 10 points pour détecter le décroché qui indique la séparation entre deux pages.

        n = 10
        nloop = 0
        while nloop < 10:
            cnt = self.get_rectangle_from_contour(c, n)
            if DEBUG:
                test = cv2.drawContours(np.copy(image), [cnt], 0, (0, 0, 255), 10)
                cv2.imwrite("0_4.png", test)
            h, w, _ = image.shape
            toppoints = dict()
            bottompoints = dict()
            # On filtre les points qui sont à peu près au centre (x/2) de l'image
            for ci in cnt:
                x, y = ci[0]
                if 0.4 * w < x and x < 0.6 * w:
                    if y < 0.2 * h:
                        toppoints[y] = x
                    elif y > 0.8 * h:
                        bottompoints[y] = x
            # Est-ce qu'on a suffisamment de points ?
            if len(toppoints) > 0 and len(bottompoints) > 0:
                break
            n = len(cnt) + 1
            nloop = nloop + 1

        if nloop == 10:
            raise Exception("Failed to found contour of the page.", "10 iterations")

        # Le point de séparation des deux pages en haut et bas
        toppoint = [toppoints[max(toppoints, key=int)], max(toppoints, key=int)]
        bottompoint = [
            bottompoints[min(bottompoints, key=int)],
            min(bottompoints, key=int),
        ]
        if bottompoint[1] == toppoint[1]:
            raise Exception("separation double page y=0", "Impossible")
        self.OUTPUT.print(
            "separation double page angle",
            np.arctan2(bottompoint[1] - toppoint[1], bottompoint[0] - toppoint[0])
            / np.pi
            * 180,
        )
        self.OUTPUT.print(
            "separation double page y=0",
            (bottompoint[0] * toppoint[1] - toppoint[0] * bottompoint[1])
            / (toppoint[1] - bottompoint[1]),
        )

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
        # On renvoie les images cropées.
        return page_gauche_0, page_droite_0

    def rotate_image(self, image, angle_deg):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle_deg, 1.0)
        result = cv2.warpAffine(
            image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR
        )
        return result

    def remets_droit_la_page(self, image, n):
        img_gauche = self.convertion_en_niveau_de_gris(image)
        # On grossit les images pour former un boudin et mieux détecter les lignes.
        eroded = cv2.erode(img_gauche, np.ones((2, 2)), iterations=7)
        if DEBUG:
            cv2.imwrite(str(n) + "_0a.png", eroded)

        # Aide à la détection des contours
        img_gauche2 = cv2.Canny(eroded, 25, 225, apertureSize=5)
        if DEBUG:
            cv2.imwrite(str(n) + "_0a2.png", img_gauche2)

        # Détection des lignes.
        list_lines = cv2.HoughLinesP(
            img_gauche2, 1, np.pi / 180, 70, minLineLength=300, maxLineGap=90
        )

        # lines contient une liste de liste de lignes.
        # Le deuxième niveau de liste ne contient toujours qu'une ligne.
        lines = map(lambda list_line: list_line[0], list_lines)

        # On filtre les lignes plutôt horizontales
        def constrait_angle(line, limit_angle=20):
            a, b, c, d = line
            angl = np.arctan2(d - b, c - a) / np.pi * 180
            return -limit_angle < angl and angl < limit_angle

        valid_lines = list(filter(constrait_angle, lines))

        if DEBUG:
            image_with_lines = image.copy()
            for line_x1, line_y1, line_x2, line_y2 in valid_lines:
                cv2.line(
                    image_with_lines,
                    (line_x1, line_y1),
                    (line_x2, line_y2),
                    (0, 0, 255),
                    1,
                )
            cv2.imwrite(str(n) + "_0b.png", image_with_lines)

        # On converti les lignes en angles
        angles = list(
            map(
                lambda line: np.arctan2(line[3] - line[1], line[2] - line[0])
                / np.pi
                * 180,
                valid_lines,
            )
        )

        # On enlève les valeurs extrêmes
        ecarttype = np.std(angles) / 2
        moyenne = np.mean(angles)
        angle_dans_ecarttype = list(
            filter(
                lambda x: moyenne - ecarttype < x and x < moyenne + ecarttype, angles
            )
        )

        rotate_angle = np.mean(angle_dans_ecarttype)
        self.OUTPUT.print("page rotation " + str(n), rotate_angle)
        # Enfin, on tourne.
        return self.rotate_image(image, rotate_angle)

    def crop_rectangle(self, image, min_x, max_x, min_y, max_y):
        return image[min_y - 10 : max_y + 10, min_x - 10 : max_x + 10]

    def isole_partie_interessante(self, image, n):
        # On force le noir et blanc
        gray = self.convertion_en_niveau_de_gris(image)
        eroded = cv2.erode(gray, np.ones((4, 4)), iterations=1)
        if DEBUG:
            cv2.imwrite(str(n) + "_1a.png", eroded)
        _, threshold = cv2.threshold(eroded, 240, 255, cv2.THRESH_BINARY)
        if DEBUG:
            cv2.imwrite(str(n) + "_1b.png", threshold)
        # On récupère le contour le plus grand.
        contours, _ = cv2.findContours(
            threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        c = max(contours, key=cv2.contourArea)
        image2222 = cv2.drawContours(image.copy(), [c], -1, (0, 0, 255), 3)
        if DEBUG:
            cv2.imwrite(str(n) + "_1c.png", image2222)

        # On garde le rectangle le plus grand.
        rect = self.get_rectangle_from_contour(c, 4)
        image22223 = cv2.drawContours(image2222, [rect], -1, (255, 0, 0), 3)
        if DEBUG:
            cv2.imwrite(str(n) + "_1d.png", image22223)

        x_crop1 = [rect[0, 0, 0], rect[1, 0, 0], rect[2, 0, 0], rect[3, 0, 0]]
        y_crop1 = [rect[0, 0, 1], rect[1, 0, 1], rect[2, 0, 1], rect[3, 0, 1]]
        x_crop1.sort()
        y_crop1.sort()

        page_gauche_0 = image[y_crop1[1] : y_crop1[2], x_crop1[1] : x_crop1[2]]
        if DEBUG:
            cv2.imwrite(str(n) + "_1e.png", page_gauche_0)

        # On enlève les bordures noirs sur le bord des pages.
        imgh, imgw, _ = page_gauche_0.shape
        min_x, min_y = imgw, imgh
        max_x = max_y = 0

        gray = self.convertion_en_niveau_de_gris(page_gauche_0)
        dilated = cv2.dilate(gray, np.ones((2, 2)))
        if DEBUG:
            cv2.imwrite(str(n) + "_1f.png", dilated)

        _, threshold = cv2.threshold(dilated, 200, 255, cv2.THRESH_BINARY)
        if DEBUG:
            cv2.imwrite(str(n) + "_1g.png", dilated)
        contours, _ = cv2.findContours(
            threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        image2222 = page_gauche_0.copy()
        for _, cnt in enumerate(contours):
            if (
                0.001 * 0.001 * imgh * imgw < cv2.contourArea(cnt)
                and cv2.contourArea(cnt) < 0.5 * imgh * 0.5 * imgw
            ):
                (x, y, w, h) = cv2.boundingRect(cnt)
                cv2.drawContours(image2222, [cnt], -1, (0, 0, 255), 3)
                min_x, max_x = min(x, min_x), max(x + w, max_x)
                min_y, max_y = min(y, min_y), max(y + h, max_y)
        if DEBUG:
            cv2.imwrite(str(n) + "_1h.png", image2222)

        self.OUTPUT.print("image " + str(n) + " crop x1", x_crop1[1] + min_x)
        self.OUTPUT.print("image " + str(n) + " crop y1", y_crop1[1] + min_y)
        self.OUTPUT.print("image " + str(n) + " crop x2", x_crop1[1] + max_x)
        self.OUTPUT.print("image " + str(n) + " crop y2", y_crop1[1] + max_y)

        return (
            page_gauche_0,
            np.maximum(min_x - 10, 0),
            np.minimum(max_x + 10, imgw - 1),
            np.maximum(min_y - 10, 0),
            np.minimum(max_y + 10, imgh - 1),
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
        self.OUTPUT.print("image " + str(n) + " dpi", dpi)

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

    def treat_file(self, filename, dict_test=None):
        img = self.charge_image(filename)

        if dict_test is None:
            self.OUTPUT = print_release.PrintRelease()
        else:
            self.OUTPUT = print_test.PrintTest(dict_test)

        image1, image2 = self.separe_double_page_en_deux(img)

        if DEBUG:
            cv2.imwrite("0_5.png", image1)
            cv2.imwrite("0_6.png", image2)

        image1a = self.remets_droit_la_page(image1, 1)
        image2a = self.remets_droit_la_page(image2, 2)

        if DEBUG:
            cv2.imwrite("1_0c.png", image1a)
            cv2.imwrite("2_0c.png", image2a)

        (
            image1a2,
            min_x1,
            max_x1,
            min_y1,
            max_y1,
            imgw1,
            imgh1,
        ) = self.isole_partie_interessante(image1a, 1)
        image1b = self.crop_rectangle(image1a2, min_x1, max_x1, min_y1, max_y1)
        (
            image2a2,
            min_x2,
            max_x2,
            min_y2,
            max_y2,
            imgw2,
            imgh2,
        ) = self.isole_partie_interessante(image2a, 2)
        image2b = self.crop_rectangle(image2a2, min_x2, max_x2, min_y2, max_y2)

        if DEBUG:
            cv2.imwrite("1_1i.png", image1b)
            cv2.imwrite("2_1i.png", image2b)

        image1c = self.recadre(
            image1b, 1, 21.0, 29.7, min_x1, max_x1, min_y1, max_y1, imgw1, imgh1
        )
        image2c = self.recadre(
            image2b, 2, 21.0, 29.7, min_x2, max_x2, min_y2, max_y2, imgw2, imgh2
        )

        cv2.imwrite(filename + "1_2.png", image1c)
        cv2.imwrite(filename + "2_2.png", image2c)
