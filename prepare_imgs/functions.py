import copy
import os
import re
from statistics import mean, median
from numpy.linalg import norm as np_norm

import cv2
import imageio.v3 as iio
import numpy as np

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from constants import CONST_CAMERA_MATRIX, CONST_DIST_COEFS
from help_funcs import sort_human

minimal_img_weight = 1000000  # Минимальный размер фотографии, байт
max_num_front_edge = 3  # До какого по счёту фото искать передний край поддона
min_img_num = 10  # Минимальное количество фотографий для склейки поддона
max_time_gap_for_pallet = 10 * 60  # Максимальная дальность поиска предыдущих фотографий поддона, с


def brightness(img):
    h, w = img.shape[:2]
    im1 = img[0:int(h / 2), int(0.05 * w):int(0.94 * w)]
    im2 = img[int(h / 2):h, int(0.05 * w):int(0.94 * w)]
    res1 = np.average(np_norm(im1, axis=2)) / np.sqrt(3)
    res2 = np.average(np_norm(im2, axis=2)) / np.sqrt(3)
    res = max(res1, res2)
    if res < 100:
        im3 = img[0:int(h / 2), int(0.05 * w):int(0.94 * w / 2)]
        im4 = img[int(h / 2):h, int(0.94 * w / 2):int(0.94 * w)]
        res3 = np.average(np_norm(im3, axis=2)) / np.sqrt(3)
        res4 = np.average(np_norm(im4, axis=2)) / np.sqrt(3)
        im5 = img[0:int(h / 2), int(0.94 * w / 2):int(0.94 * w)]
        im6 = img[int(h / 2):h, int(0.05 * w):int(0.94 * w / 2)]
        res5 = np.average(np_norm(im5, axis=2)) / np.sqrt(3)
        res6 = np.average(np_norm(im6, axis=2)) / np.sqrt(3)
        res = min(res3, res4, res5, res6)
    else:
        im3 = img[0:int(h / 2), int(0.05 * w):int(0.94 * w / 2)]
        im4 = img[int(h / 2):h, int(0.94 * w / 2):int(0.94 * w)]
        res3 = np.average(np_norm(im3, axis=2)) / np.sqrt(3)
        res4 = np.average(np_norm(im4, axis=2)) / np.sqrt(3)
        im5 = img[0:int(h / 2), int(0.94 * w / 2):int(0.94 * w)]
        im6 = img[int(h / 2):h, int(0.05 * w):int(0.94 * w / 2)]
        res5 = np.average(np_norm(im5, axis=2)) / np.sqrt(3)
        res6 = np.average(np_norm(im6, axis=2)) / np.sqrt(3)
        res = max(res3, res4, res5, res6)
    return res


def remove_distortsion(img):
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(CONST_CAMERA_MATRIX, CONST_DIST_COEFS, (w, h), 1, (w, h))

    dst = cv2.undistort(img, CONST_CAMERA_MATRIX, CONST_DIST_COEFS, None, newcameramtx)
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    # crop and save the image
    x, y, w, h = roi
    return dst[y:y + h, x:x + w]


def crop(image):
    y_nonzero, x_nonzero, _ = np.nonzero(image)
    return image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]


def rotate_image(mat, angle=180.5):

    height, width = mat.shape[:2]  # image shape has 3 dimensions
    image_center = (
        width / 2,
        height / 2)  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


def detect_border(img_path):  # Поиск правой и левой границ поддона
    input_img = iio.imread(img_path)
    input_img = rotate_image(crop(remove_distortsion(input_img)))
    left_thresh_limit_1 = 190
    left_thresh_limit_2 = 190
    min_line_length_left = 0.0123
    right_thresh_limit_1 = 200
    right_thresh_limit_2 = 210
    min_line_length_right = 0.0123
    h, w, channels = input_img.shape
    white_bright = brightness(input_img)
    if white_bright < 180:
        left_thresh_limit_1 = 190
        left_thresh_limit_2 = 190
        right_thresh_limit_1 = 180
        right_thresh_limit_2 = 180
    if white_bright < 170:
        left_thresh_limit_1 = 180
        left_thresh_limit_2 = 180
        right_thresh_limit_1 = 170
        right_thresh_limit_2 = 170
        min_line_length_left = 0.01
        min_line_length_right = 0.01
    if white_bright < 160:
        left_thresh_limit_1 = 170
        left_thresh_limit_2 = 190
        right_thresh_limit_1 = 160
        right_thresh_limit_2 = 160
        min_line_length_left = 0.01
        min_line_length_right = 0.01
    if white_bright < 150:
        left_thresh_limit_1 = 170
        left_thresh_limit_2 = 190
        right_thresh_limit_1 = 150
        right_thresh_limit_2 = 160
        min_line_length_left = 0.01
        min_line_length_right = 0.01
    if white_bright < 140:
        left_thresh_limit_1 = 190
        left_thresh_limit_2 = 190
        right_thresh_limit_1 = 150
        right_thresh_limit_2 = 150
        min_line_length_left = 0.01
        min_line_length_right = 0.01
    if white_bright < 120:
        left_thresh_limit_1 = 100
        left_thresh_limit_2 = 100
        right_thresh_limit_1 = 150
        right_thresh_limit_2 = 150
        min_line_length_left = 0.01
        min_line_length_right = 0.01
    if white_bright < 60:
        left_thresh_limit_1 = 70
        left_thresh_limit_2 = 90
        right_thresh_limit_1 = 50
        right_thresh_limit_2 = 50
        min_line_length_left = 0.04
        min_line_length_right = 0.04
    if white_bright < 40:
        left_thresh_limit_1 = 60
        left_thresh_limit_2 = 70
        right_thresh_limit_1 = 10
        right_thresh_limit_2 = 10
        min_line_length_left = 0.05
        min_line_length_right = 0.04
    left_b_coor = [int(0.07 * w), int(0.1 * w)]
    right_b_coor = [int(0.9 * w), int(0.944 * w)]
    left_border = input_img[0:h, left_b_coor[0]:left_b_coor[1]]
    right_border = input_img[0:h, right_b_coor[0]:right_b_coor[1]]
    del input_img
    # input_img = input_img[10:h, 0:w]
    gray_left = cv2.cvtColor(left_border, cv2.COLOR_BGR2GRAY)
    ret_left, thresh_left = cv2.threshold(gray_left, left_thresh_limit_1, 255, cv2.THRESH_BINARY)
    gray_right = cv2.cvtColor(right_border, cv2.COLOR_BGR2GRAY)
    ret_right, thresh_right = cv2.threshold(gray_right, right_thresh_limit_1, 255, cv2.THRESH_BINARY)
    # cv2.namedWindow('thresh', cv2.WINDOW_NORMAL)
    # cv2.imshow('thresh', thresh_left)
    # cv2.waitKey(0)
    # cv2.imshow('thresh', thresh_right)
    # cv2.waitKey(0)
    edges_left = rotate_image(cv2.Canny(thresh_left, 100, 200), 1)
    edges_right = cv2.Canny(thresh_right, 100, 200)
    del left_border
    del right_border
    del gray_left
    del gray_right
    # cv2.imshow('thresh', edges_left)
    # cv2.waitKey(0)
    # cv2.imshow('thresh', edges_right)
    # cv2.waitKey(0)
    minLineLength_left = int(min_line_length_left * h)  # 15/1228
    minLineLength_right = int(min_line_length_right * h)  # 15/1228
    maxLineGap = int(0.0035 * w)  # 5/1531
    lines_right = cv2.HoughLinesP(edges_right, 1, np.pi / 180, right_thresh_limit_2, minLineLength=minLineLength_right,
                                  maxLineGap=maxLineGap)
    lines_left = cv2.HoughLinesP(edges_left, 1, np.pi / 180, left_thresh_limit_2, minLineLength=minLineLength_left,
                                 maxLineGap=maxLineGap)
    vertical_lines = []
    vertical_lines_left = []
    vertical_lines_right = []
    count_left_lines = 0
    count_right_lines = 0
    # cv2.namedWindow('FLD', cv2.WINDOW_NORMAL)
    if lines_left is not None:
        for line in lines_left:
            for x1, y1, x2, y2 in line:
                # #print('x:', x1, x2)
                # #print('y:', y1, y2)
                # cv2.line(left_border, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # cv2.imshow("FLD", left_border)
                # cv2.waitKey(0)
                if abs(y2 - y1) >= minLineLength_left and abs(x2 - x1) <= int(0.00653 * w):  # 10/1531
                    count_left_lines += 1
                    vertical_lines.append(min([x1 + left_b_coor[0], x2 + left_b_coor[0]]))
                    vertical_lines_left.append(mean([y1, y2]))
                    # cv2.line(input_img, (x1+left_b_coor[0], y1), (x2+left_b_coor[0], y2), (255, 0, 0), 2)
    if lines_right is not None:
        for line in lines_right:
            for x1, y1, x2, y2 in line:
                # #print('x:', x1, x2)
                # #print('y:', y1, y2)
                # cv2.line(right_border, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # cv2.imshow("FLD", right_border)
                # cv2.waitKey(0)
                if abs(y2 - y1) >= minLineLength_right and abs(x2 - x1) <= int(0.00653 * w):  # 10/1531
                    count_right_lines += 1
                    vertical_lines.append(max([x1 + right_b_coor[0], x2 + right_b_coor[0]]))
                    vertical_lines_right.append(mean([y1, y2]))
                    # cv2.line(input_img, (x1+right_b_coor[0], y1), (x2+right_b_coor[0], y2), (255, 0, 0), 2)
    # #print(count_horizontal_lines)

    #
    # cv2.imshow("FLD", input_img)
    # cv2.waitKey(0)
    if count_left_lines != 0 and count_right_lines != 0:
        if len(vertical_lines_right) > 2 and len(vertical_lines_left) > 2:
            min_y = min(vertical_lines_left)
            if min(vertical_lines_right) > min_y:
                min_y = min(vertical_lines_right)
            max_y = max(vertical_lines_left)
            if max(vertical_lines_right) < max_y:
                max_y = max(vertical_lines_right)
            vertical_lines_left_tmp = list(filter(lambda i: min_y < i < max_y, vertical_lines_left))
            if len(vertical_lines_left_tmp) != 0:
                vertical_lines_left = vertical_lines_left_tmp
        return vertical_lines, vertical_lines_left, white_bright
    else:
        return None, [], white_bright


def detect_edges(img_path, left_lines, v_lines, white_bright):
    # print(img_path)
    input_img = iio.imread(img_path)
    input_img = rotate_image(crop(remove_distortsion(input_img)))
    thresh_limit_1 = 210
    thresh_limit_2 = 230
    lineLength = 0.06
    secondary_filter = True

    h, w, channels = input_img.shape
    input_img_h = input_img[int(0.004 * h):h, min(v_lines):max(v_lines)]  # 5/1228
    if white_bright > 170:
        thresh_limit_1 = 215
        thresh_limit_2 = 230
        lineLength = 0.07
    if white_bright > 180:
        thresh_limit_1 = 215
        thresh_limit_2 = 240
        lineLength = 0.09
    if white_bright > 200:
        thresh_limit_1 = 220
        thresh_limit_2 = 240
        lineLength = 0.1
    if white_bright < 146:
        thresh_limit_1 = 210
        thresh_limit_2 = 210
        lineLength = 0.06
        secondary_filter = False
    if white_bright < 120:
        thresh_limit_1 = 160
        thresh_limit_2 = 200
        lineLength = 0.05
        secondary_filter = False
    if white_bright < 110:
        thresh_limit_1 = 120
        thresh_limit_2 = 200
        lineLength = 0.05
        secondary_filter = False
    if white_bright < 60:
        thresh_limit_1 = 80
        thresh_limit_2 = 140
        lineLength = 0.07
        secondary_filter = False
    if white_bright < 40:
        thresh_limit_1 = 70
        thresh_limit_2 = 140
        lineLength = 0.09
        secondary_filter = False

    del input_img
    gray = cv2.cvtColor(input_img_h, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, thresh_limit_1, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(thresh, 100, 200)
    minLineLength = int(lineLength * w)  # 58/1531
    maxLineGap = int(0.0081 * h)  # 10/1228
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, thresh_limit_2, minLineLength=minLineLength, maxLineGap=maxLineGap)
    horizontal_lines = []
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                # #print('x:', x1, x2)
                # #print('y:', y1, y2)
                # cv2.line(input_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # cv2.imshow("FLD", input_img)
                # cv2.waitKey(0)
                if abs(x2 - x1) >= minLineLength and abs(y2 - y1) <= int(0.01628 * h):  # 20/1228
                    horizontal_lines.extend([y1 + int(0.004 * h), y2 + int(0.004 * h)])
                    # cv2.line(input_img_h, (x1, y1), (x2, y2), (255, 0, 0), 2)
    if secondary_filter:
        if len(horizontal_lines) != 0:
            horizontal_lines = list(filter(lambda y: abs(y - median(horizontal_lines)) < 15, horizontal_lines))
    del input_img_h
    if len(horizontal_lines) >= 2 and len(left_lines) != 0:
        if (max(horizontal_lines) - min(left_lines) > 10) and (max(left_lines) - min(horizontal_lines) > 10):
            # print(img_path, 'has front edge')
            # cv2.imwrite('E:/GreenStuff/GE_image_stitching/photos/correction/edges/both/' + img_path.split('/')[-1],
            #             input_img)
            result_edge = 'Both'
        elif max(horizontal_lines) - min(left_lines) > 10:
            # print(img_path, 'has rear edge')
            # cv2.imwrite('E:/GreenStuff/GE_image_stitching/photos/correction/edges/front/' + img_path.split('/')[-1],
            #             input_img)
            result_edge = 'Front'
        elif max(left_lines) - min(horizontal_lines) > 10:
            # print(img_path, 'has both edges')
            result_edge = 'Rear'
        return result_edge, horizontal_lines
    else:
        # cv2.imwrite('E:/GreenStuff/GE_image_stitching/photos/correction/edges/' + img_path.split('/')[-1],
        #             input_img)
        return None, None


def split_arr(arr, edges_info, final_split_arrs=None):  # Разбиение массива (если в массив попало более 1 поддона)
    not_1_pallet = False
    has_front_edge = False
    has_rear_edge = False
    has_both_edges = False
    if final_split_arrs is None:
        final_split_arrs = []
    for path in arr:
        result_edge, info = edges_info[path].values()
        if result_edge == 'Both' or (result_edge == 'Front' and has_rear_edge):
            not_1_pallet = True
            # print('NOT_1_PALLET')
            break
        if result_edge == 'Rear':
            has_rear_edge = True
    if not_1_pallet:  # Если в массиве есть фото с двумя поддонами, или если появился передний край поддона после фото с задним краем поддона, то там больше одного поддона
        last_img_path = ''
        for img_path in arr:
            result_edge, info = edges_info[img_path].values()
            if (result_edge == 'Rear' or result_edge == 'Both') and (arr.index(
                    img_path) >= max_num_front_edge):  # Если появился задний край или 2 поддона, то это конец поддона
                last_img_path = img_path
                # print('FOUND_LAST_IMG_PATH',last_img_path)
                break
        if last_img_path != '':
            first_image_in_second_dir = ''
            for img_path in arr[arr.index(last_img_path) + 1:]:
                result_edge, info = edges_info[img_path].values()
                # print('result_edge',result_edge)
                if result_edge == 'Front' or result_edge == 'Both':
                    first_image_in_second_dir = img_path
                    # print('FIND first_image_in_second_dir', first_image_in_second_dir)
                elif result_edge is None:
                    # print('FIND FINAL first_image_in_second_dir', first_image_in_second_dir)
                    break
            if first_image_in_second_dir != '':
                new_arr = arr[arr.index(first_image_in_second_dir):]
            else:
                new_arr = arr[arr.index(last_img_path) + 1:]
            old_arr_stripped = arr[0:arr.index(last_img_path) + 1]
            # print('OLD_ARR=', old_arr_stripped)
            # print('NEW_ARR=', new_arr)
            if len(final_split_arrs) == 0:
                final_split_arrs.append(old_arr_stripped)
                final_split_arrs.extend(split_arr(new_arr, edges_info, final_split_arrs))
            else:
                final_split_arrs.append(old_arr_stripped)
                final_split_arrs = split_arr(new_arr, edges_info, final_split_arrs)
            return final_split_arrs
        else:
            return [arr]
    else:
        return [arr]


def find_front_edge(f_final, our_key, our_first_key, edges_info):
    nearest_key = ''
    min_distance = 9999999
    for ok_key in f_final:
        if min_distance > (int(our_key.split('_')[0]) - int(ok_key.split('_')[0])) and (
                int(our_key.split('_')[0]) - int(ok_key.split('_')[0])) > 0:
            min_distance = int(our_key.split('_')[0]) - int(ok_key.split('_')[0])
            nearest_key = ok_key
        if int(our_key.split('_')[0]) - int(ok_key.split('_')[0]) <= 0:
            break
    if nearest_key != '':
        # print('!nearest_key|||=', nearest_key)
        for final_key in f_final:
            if final_key.find(nearest_key) != -1:
                nearest_key = final_key
                # print('!!!!!!!!!!')
                # print('nearest_key=', nearest_key)
        if int(nearest_key.split('_')[0]) - int(our_first_key.split('_')[
                                                    0]) > max_time_gap_for_pallet:  # Если предыдущее фото сделано слишком давно, то это не может быть наш поддон
            return None
        has_child_edge = False
        last_child_edge = None
        for img_path in f_final[nearest_key]:  # Ищем в ней край поддона
            result_edge, info = edges_info[img_path].values()
            if result_edge == 'Front' or result_edge == 'Both' or result_edge == 'Rear':
                last_child_edge = result_edge
        if last_child_edge == 'Front' or last_child_edge == 'Both':
            has_child_edge = True
        if has_child_edge:
            return nearest_key
        else:
            return find_front_edge(f_final, nearest_key, our_first_key, edges_info)
    else:
        return None


def create_final_dict(need_key_path):
    raw_folder = need_key_path[:need_key_path.rfind('/') + 1]
    need_key = need_key_path.split('/')[-1].split('_')[0]
    f_full_paths = []
    for path in os.listdir(raw_folder):
        if len(re.findall(r'_(\d)+\.png', path)) != 0 and path.split('.png')[-1] == '' and 0 <= (
                int(need_key) - int(path.split('/')[-1].split('_')[0])) <= max_time_gap_for_pallet and os.path.getsize(
            raw_folder + path) > minimal_img_weight:  # Выбираем все пути, оканчивающиеся на "_[число].png", на расстоянии не более max_time_gap_for_pallet и размером > minimal_img_weight
            f_full_paths.append(raw_folder + path)
    f_full = dict.fromkeys([key.split('/')[-1].split('_')[0] for key in f_full_paths],
                           [])  # Словарь с фото размером > minimal_img_weight
    f_ok = dict()  # Словарь папок с фото размером > minimal_img_weight и при условии, что на фото есть правая и левая границы
    borders_info = dict()
    edges_info = dict()
    left_lines_info = dict()
    white_bright_info = dict()
    # граница.
    for key in f_full:
        f_full[key] = list(filter(lambda path: path.split('/')[-1].split('_')[0] == key, f_full_paths))
        f_full[key] = sort_human(f_full[key])
        tmp_arr = []
        for path in f_full[key]:
            borders, left_lines, white_bright = detect_border(path)
            if borders is not None:
                tmp_arr.append(path)
                borders_info[path] = borders
                left_lines_info[path] = left_lines
                white_bright_info[path] = white_bright
        f_ok[key] = tmp_arr
        # f_ok[key] = list(filter(lambda img_path: detect_border(img_path) is not None, f_full[key]))
        if len(f_ok[key]) == 0:
            del f_ok[key]
        # if len(f_full[key]) >= minimal_img_num:
        #     f_ok[key] = f_full[key]
    # print(borders_info)
    f_ok = dict(sorted(f_ok.items()))
    # print(f_ok)
    f_final = copy.deepcopy(f_ok)  # Финальный словарь для склеивания
    for ok_key in f_ok:
        not_1_pallet = False
        has_rear_edge = False
        for img_path in f_ok[ok_key]:
            result_edge, info = detect_edges(img_path, left_lines_info[img_path], borders_info[img_path],
                                             white_bright_info[img_path])
            edges_info[img_path] = {'result_edge': result_edge, 'info': info}
            if result_edge == 'Both' or (result_edge == 'Front' and has_rear_edge):
                # print('*******HAS MORE THAN 1 PALLET', ok_key)
                not_1_pallet = True
            if result_edge == 'Rear':
                # print('*****HAS REAR EDGE', ok_key)
                has_rear_edge = True
        if not_1_pallet:
            count_new = 0
            for arr in split_arr(f_ok[ok_key], edges_info):
                if len(arr) != 0:
                    f_final[ok_key + '_new' * count_new] = arr
                    count_new += 1

        has_own_edge = False
        for img_path in f_ok[ok_key]:  # Смотрим первые max_num_front_edge фотографии
            result_edge, info = edges_info[img_path].values()
            if result_edge == 'Rear':
                break
            if result_edge == 'Front':
                has_own_edge = True
                break
        if not has_own_edge:  # Если среди них нет переднего края поддона, то ищем передний край в предыдущих ключах
            nearest_key = find_front_edge(f_final, ok_key, our_first_key=ok_key, edges_info=edges_info)
            # print('FINAL_NEAREST_KEY=', nearest_key)
            if nearest_key is not None:
                found_start_key = False
                del_keys = []
                tmp_arr = []
                for final_key in sorted(list(f_final.keys())):
                    # print('final_key', final_key)
                    # print('nearest_key', nearest_key)
                    if nearest_key == final_key:
                        # print('FOUND_START_KEY')
                        found_start_key = True
                    # print('ok_key', ok_key)
                    if ok_key == final_key:
                        # print('FOUND_END_KEY')
                        tmp_arr.extend(f_final[final_key])
                        # print('tmp_arr', tmp_arr)
                        break
                    if found_start_key:
                        del_keys.append(final_key)
                        tmp_arr.extend(f_final[final_key])
                        # print('f_final[final_key]', f_final[final_key])
                        # print('tmp_arr',tmp_arr)
                # print('tmp_arr_END', tmp_arr)
                f_final[ok_key] = tmp_arr
                for del_key in del_keys:
                    del f_final[del_key]
    # print('***************************************')
    debug_info = dict.fromkeys(f_final.keys())
    for final_key in f_final:
        # print('FINAL_KEY', final_key)
        debug_info[final_key] = {'skips': [], 'front_edge': [], 'rear_edge': [], 'borders': {}}
        first_img_path = ''
        res_info = ''
        final_res_edge = None
        for img_path in f_final[final_key]:
            result_edge, info = edges_info[img_path].values()
            if result_edge == 'Front' or result_edge == 'Both':
                final_res_edge = result_edge
                first_img_path = img_path
                res_info = info
            else:
                if res_info != '':
                    if final_res_edge == 'Both':
                        debug_info[final_key]['front_edge'].extend([first_img_path, [min(res_info)]])
                    else:
                        debug_info[final_key]['front_edge'].extend([first_img_path, res_info])
                break
        if first_img_path != '':
            f_final[final_key] = f_final[final_key][f_final[final_key].index(first_img_path):]
        last_img_path = ''
        for img_path in f_final[final_key]:  # Начиная с minimal_img_num, ищем задний край
            result_edge, info = edges_info[img_path].values()
            if result_edge == 'Rear':
                last_img_path = img_path
                debug_info[final_key]['rear_edge'].extend([last_img_path, info])
                break
        if last_img_path != '':
            f_final[final_key] = f_final[final_key][0:f_final[final_key].index(last_img_path) + 1]

        previous_num = int(f_final[final_key][0].split('_')[-1].split('.')[0])
        for img_path in f_final[final_key]:
            debug_info[final_key]['borders'][img_path] = borders_info[img_path]
            if int(img_path.split('_')[-1].split('.')[0]) - previous_num > 1:
                debug_info[final_key]['skips'].append(img_path)
            previous_num = int(img_path.split('_')[-1].split('.')[0])
        # print(final_key)
        # print(f_final[final_key])

    # print(len(f_full))
    # print(len(f_ok))
    # print(len(f_final))
    # print('END_PREPARING')
    return f_final, debug_info


def list_4_remove(need_key_path):
    f_all_paths = []
    raw_folder = need_key_path[:need_key_path.rfind('/') + 1]
    need_key = need_key_path.split('/')[-1].split('_')[0]
    for path in os.listdir(raw_folder):
        if len(re.findall(r'_(\d)+\.png', path)) != 0 and max_time_gap_for_pallet < (
                int(need_key) - int(path.split('/')[-1].split('_')[0])):
            f_all_paths.append(raw_folder + path)
    return f_all_paths
