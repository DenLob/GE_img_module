import gc
import cv2
import imageio.v3 as iio
from PIL import ImageFile
import imutils
import numpy as np

from constants import CONST_RESULT_FOLDER
from prepare_imgs.functions import rotate_image, crop, remove_distortsion

ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    from cv2.cuda import SURF_CUDA_create

    feature_extractor = 'surf'  # one of 'sift', 'surf', 'brisk', 'orb'
    # feature_extractor = 'sift'
except ImportError:
    feature_extractor = 'sift'
feature_matching = 'knn'


def detectAndDescribe(image, method=None):
    """
    Compute key points and feature descriptors using an specific method
    """

    assert method is not None, "You need to define a feature detection method. Values are: 'sift', 'surf'"

    # detect and extract features from the image
    if method == 'sift':
        descriptor = cv2.SIFT_create()
    elif method == 'surf':
        descriptor = cv2.cuda.SURF_CUDA_create(100)
    elif method == 'brisk':
        descriptor = cv2.BRISK_create()
    elif method == 'orb':
        descriptor = cv2.ORB_create()

    # get keypoints and descriptors
    if method == 'surf':
        (kps, features) = descriptor.detectWithDescriptors(image, None)
        return (kps, features, descriptor)
    else:
        (kps, features) = descriptor.detectAndCompute(image, None)
        return (kps, features, None)


def createMatcher(method, crossCheck):
    "Create and return a Matcher Object"

    if method == 'surf':
        bf = cv2.cuda.DescriptorMatcher_createBFMatcher(cv2.NORM_L2)
    if method == 'sift':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
    elif method == 'orb' or method == 'brisk':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
    return bf


def matchKeyPointsBF(featuresA, featuresB, method, logger):
    bf = createMatcher(method, crossCheck=True)

    # Match descriptors.
    best_matches = bf.match(featuresA, featuresB)

    # Sort the features in order of distance.
    # The points with small distance (more similarity) are ordered first in the vector
    rawMatches = sorted(best_matches, key=lambda x: x.distance)
    logger.debug("Raw matches (Brute force): " + str(len(rawMatches)))
    return rawMatches


def matchKeyPointsKNN(featuresA, featuresB, ratio, method, logger):
    bf = createMatcher(method, crossCheck=False)
    # compute the raw matches and initialize the list of actual matches
    rawMatches = bf.knnMatch(featuresA, featuresB, 2)
    logger.debug("Raw matches (knn): " + str(len(rawMatches)))
    matches = []

    # loop over the raw matches
    for m, n in rawMatches:
        # ensure the distance is within a certain ratio of each
        # other (i.e. Lowe's ratio test)
        if m.distance < n.distance * ratio:
            matches.append(m)
    return matches


def getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh, img_path1, img_path2, logger,
                  descriptor=None):
    # convert the keypoints to numpy arrays

    if descriptor is None:
        kpsA = np.float32([kp.pt for kp in kpsA])
        kpsB = np.float32([kp.pt for kp in kpsB])
    else:
        kpsA = np.float32([kp.pt for kp in cv2.cuda_SURF_CUDA.downloadKeypoints(descriptor, kpsA)])
        kpsB = np.float32([kp.pt for kp in cv2.cuda_SURF_CUDA.downloadKeypoints(descriptor, kpsB)])

    if len(matches) < 500:
        logger.error('TOO_LITTLE_MATCHES ' + str(len(matches)) + ' between ' + img_path1.split('/')[
            -1] + ' and ' + img_path2.split('/')[-1])
    if len(matches) > 4:
        # construct the two sets of points
        ptsA = np.float32([kpsA[m.queryIdx] for m in matches])
        ptsB = np.float32([kpsB[m.trainIdx] for m in matches])
        # estimate the homography between the sets of points
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                                         reprojThresh)
        return (matches, H, status, ptsA, ptsB)
    else:
        return None


def im_stitching(imgs_pathes, new_name, debug_info, logger):
    global feature_matching
    stitched_img = None
    if feature_extractor == 'surf':
        trainImg_gpu = cv2.cuda_GpuMat()
        queryImg_gpu = cv2.cuda_GpuMat()
    # if os.path.exists(CONST_RESULT_FOLDER +
    #                   'res_' + new_name + '.' + imgs_pathes[-1].split('.')[-1]):
    #     return
    previous_img = rotate_image(crop(remove_distortsion(iio.imread(imgs_pathes[0]))), 180)
    if len(debug_info[new_name]['front_edge']) != 0 and imgs_pathes[0] == debug_info[new_name]['front_edge'][0]:
        previous_img = previous_img[0:max(debug_info[new_name]['front_edge'][1]),
                       min(debug_info[new_name]['borders'][imgs_pathes[0]]): max(
                           debug_info[new_name]['borders'][imgs_pathes[0]])]
    else:
        logger.error('NO FRONT EDGE ' + imgs_pathes[0].split('/')[-1])
        previous_img = previous_img[0:previous_img.shape[0], min(debug_info[new_name]['borders'][imgs_pathes[0]]): max(
            debug_info[new_name]['borders'][imgs_pathes[0]])]
    for img_path in imgs_pathes[1:]:
        logger.debug('Opening ' + img_path)
        queryImg = rotate_image(crop(remove_distortsion(iio.imread(img_path))), 180)
        h, w, channels = queryImg.shape
        if len(debug_info[new_name]['rear_edge']) != 0 and img_path == debug_info[new_name]['rear_edge'][0]:
            queryImg = queryImg[min(debug_info[new_name]['rear_edge'][1]):h,
                       min(debug_info[new_name]['borders'][img_path]): max(debug_info[new_name]['borders'][img_path])]
        else:
            queryImg = queryImg[0:h,
                       min(debug_info[new_name]['borders'][img_path]): max(debug_info[new_name]['borders'][img_path])]
        if feature_extractor == 'surf':
            trainImg_gpu.upload(previous_img)  # unnormal oriented photos
            trainImg_gray = cv2.cuda.cvtColor(trainImg_gpu, cv2.COLOR_RGB2GRAY)
            queryImg_gpu.upload(queryImg)
            queryImg_gray = cv2.cuda.cvtColor(queryImg_gpu, cv2.COLOR_RGB2GRAY)
        else:
            trainImg_gray = cv2.cvtColor(previous_img, cv2.COLOR_RGB2GRAY)
            queryImg_gray = cv2.cvtColor(queryImg, cv2.COLOR_RGB2GRAY)

        kpsA, featuresA, descriptor = detectAndDescribe(trainImg_gray, method=feature_extractor)
        kpsB, featuresB, descriptor = detectAndDescribe(queryImg_gray, method=feature_extractor)
        if feature_extractor == 'sift':
            del trainImg_gray
            del queryImg_gray
            gc.collect()
        logger.debug("Using: {} feature matcher".format(feature_matching))
        if feature_matching == 'bf':
            matches = matchKeyPointsBF(featuresA, featuresB, method=feature_extractor, logger=logger)
        elif feature_matching == 'knn':
            matches = matchKeyPointsKNN(featuresA, featuresB, ratio=0.75, method=feature_extractor, logger=logger)
        M = getHomography(kpsA, kpsB, featuresA, featuresB, matches,
                          img_path1=imgs_pathes[imgs_pathes.index(img_path) - 1], img_path2=img_path,
                          descriptor=descriptor, reprojThresh=4, logger=logger)
        if M is None:
            logger.error("M is None!")
            feature_matching_old = feature_matching
            if feature_matching == 'knn':
                feature_matching = 'bf'
            else:
                feature_matching = 'knn'
            logger.debug("Using: {} feature matcher".format(feature_matching))
            if feature_matching == 'bf':
                matches = matchKeyPointsBF(featuresA, featuresB, method=feature_extractor, logger=logger)
            elif feature_matching == 'knn':
                matches = matchKeyPointsKNN(featuresA, featuresB, ratio=0.75, method=feature_extractor, logger=logger)
            M = getHomography(kpsA, kpsB, featuresA, featuresB, matches,
                              img_path1=imgs_pathes[imgs_pathes.index(img_path) - 1], img_path2=img_path,
                              descriptor=descriptor, reprojThresh=4, logger=logger)
            feature_matching = feature_matching_old
        if M is None:
            logger.error("M is None!")
            if feature_extractor == 'surf':
                return
            else:
                if stitched_img is None:
                    h1, w1 = queryImg.shape[:2]
                    h2, w2 = previous_img.shape[:2]

                    # create empty matrix
                    stitched_img = np.zeros((h1+ h2, max(w1,w2), 3), np.uint8)

                    # combine 2 images
                    stitched_img[:h1, :w1, :3] = queryImg
                    stitched_img[h1:h1+h2, :w2, :3] = previous_img
                else:
                    h1, w1 = queryImg.shape[:2]
                    h2, w2 = stitched_img.shape[:2]

                    # create empty matrix
                    stitched_img_tmp = np.zeros((h1 + h2, max(w1, w2), 3), np.uint8)

                    # combine 2 images
                    stitched_img_tmp[:h1, :w1, :3] = queryImg
                    stitched_img_tmp[h1:h1 + h2, :w2, :3] = stitched_img
                    stitched_img = stitched_img_tmp
                    del stitched_img_tmp
                    gc.collect()
        if M is not None:
            (matches, H, status, ptsA, ptsB) = M
            # print(H)
            if H is None:
                logger.error("H is None!")
                # print(matches, H, status)
                return
            # Apply panorama correction
            if stitched_img is None:
                width = previous_img.shape[1] + queryImg.shape[1]
                height = previous_img.shape[0] + queryImg.shape[0]
            else:
                if feature_extractor == 'surf':
                    width = stitched_img.size()[0] + queryImg.shape[1]
                    height = stitched_img.size()[1] + queryImg.shape[0]
                else:
                    width = stitched_img.shape[1] + queryImg.shape[1]
                    height = stitched_img.shape[0] + queryImg.shape[0]

            if feature_extractor == 'surf':
                if stitched_img is None:
                    stitched_img = cv2.cuda.warpPerspective(trainImg_gpu, H, (width, height))
                else:
                    stitched_img = cv2.cuda.warpPerspective(stitched_img, H, (width, height))
                stitched_img = stitched_img.download()
            else:
                if stitched_img is None:
                    stitched_img = cv2.warpPerspective(previous_img, H, (width, height))
                else:
                    stitched_img = cv2.warpPerspective(stitched_img, H, (width, height))
            stitched_img[0:queryImg.shape[0], 0:queryImg.shape[1]] = queryImg
        # transform the panorama image to grayscale and threshold it
        gray = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

        # Finds contours from the binary image
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # get the maximum contour area
        c = max(cnts, key=cv2.contourArea)

        # get a bbox from the contour area
        (x, y, w, h) = cv2.boundingRect(c)

        # crop the image to the bbox coordinates
        if img_path == imgs_pathes[-1]:
            # print('saved')
            cv2.imwrite(CONST_RESULT_FOLDER +
                        'res_' + new_name + '.' + imgs_pathes[-1].split('.')[-1],
                        stitched_img[y:y + h, x:x + w])
            if feature_extractor == 'sift':
                del stitched_img
                del queryImg
                gc.collect()
            logger.info('Img saved as ' + CONST_RESULT_FOLDER +
                        'res_' + new_name + '.' + imgs_pathes[-1].split('.')[-1])
            return CONST_RESULT_FOLDER + 'res_' + new_name + '.' + imgs_pathes[-1].split('.')[-1]
        else:
            logger.debug('Proceed ' + img_path)
            previous_img = queryImg
            stitched_img = stitched_img[y:y + h, x:x + w]
            if feature_extractor == 'surf':
                tmp = stitched_img
                stitched_img = cv2.cuda_GpuMat()
                stitched_img.upload(tmp)
