import cv2 as cv
import numpy as np


def count_plants(img_path, logger):
    res_centers = []
    image = cv.imread(img_path)
    imageHSV = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # будет плавать в зависимости от освещения, нужно корректировать
    low_thresh = (30, 60, 100)
    high_thresh = (70, 255, 250)

    # можно поэкспериментировать с параметрами чтобы добиться лучшей фильтрации
    filtered = cv.inRange(imageHSV, low_thresh, high_thresh)
    kernel = np.ones((7, 7), np.uint8)
    filtered = cv.morphologyEx(filtered, cv.MORPH_OPEN, kernel)
    filtered = cv.morphologyEx(filtered, cv.MORPH_CLOSE, kernel)
    kernel = np.ones((3, 3), np.uint8)
    filtered = cv.morphologyEx(filtered, cv.MORPH_DILATE, kernel)

    conts, _, = cv.findContours(filtered, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    if len(conts) == 0:
        del image
        del imageHSV
        del filtered
        logger.info(str(0) + ' plants on ' + img_path)
        return []
    area = 0
    for cont in conts:
        area += cv.contourArea(cont)
    averageArea = area / len(conts)

    plants = 0

    for cont in conts:
        contArea = cv.contourArea(cont)
        if contArea > averageArea * 0.5:
            plants += 1
            (x_mean, y_mean), radius = cv.minEnclosingCircle(cont)
            res_centers.append([int(x_mean), int(y_mean)])
            # cv.circle(image, (int(x_mean), int(y_mean)), 10, (255, 255, 255), -1)
            # cv.drawContours(image, cont, -1, (255, 0, 255), 10)

        if contArea > averageArea * 2.8:
            rect = cv.minAreaRect(cont)
            rect_area = rect[1][0] * rect[1][1]
            box = cv.boxPoints(rect)
            box = np.int0(box)

            (x, y), radius = cv.minEnclosingCircle(cont)
            center = (int(x), int(y))
            radius = int(radius)
            circle_area = 3.14 * radius ** 2

            relative_area = circle_area / rect_area

            if relative_area > 1.6:
                plants += 1
                res_centers.append([int(x), int(y)])
                # cv.drawContours(image, cont, -1, (255, 0, 0), 10)

    logger.info(str(plants) + ' plants on ' + img_path)
    # cv.imwrite(img_path[:img_path.rfind('.')] + '_count_' + str(plants) + img_path[img_path.find('.'):], image)
    del image
    del imageHSV
    del filtered
    return res_centers
