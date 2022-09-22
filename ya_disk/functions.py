import json
import os
import time

import requests
import yadisk

from help_funcs import path2str_date
from ya_disk.ya_consts import ya_token, upload_dir


def retry_connection(func, logger, args=None):
    result = None
    try:
        if args is not None:
            if len(list(filter(lambda x: callable(x), args))) == 0:
                result = func(*args)
            else:
                result = func(args[0](*args[1]))
        else:
            result = func()
    except requests.exceptions.ConnectionError as e:
        logger.error(e)
        #print('Lost connection')
        time.sleep(3)
        result = retry_connection(func, logger, args=args)
    except yadisk.exceptions.ForbiddenError as e:
        #print('HTTP-403')
        logger.error(e)                                         # Добавить вывод в сигнализатор об ошибках
        time.sleep(3)
        result = retry_connection(func, logger, args=args)
    except yadisk.exceptions.UnauthorizedError as e:
        #print('Change token')
        logger.error(e)
        while True:
            pass                                                # Добавить вывод в сигнализатор об ошибках
    except yadisk.exceptions.PathExistsError as e:
        #print(e)
        logger.error(e)
    except yadisk.exceptions.LockedError as e:
        logger.error(e)                                         # Добавить вывод в сигнализатор об ошибках
        while True:
            pass
    except Exception as e:
        logger.error(e)                                         # Добавить вывод в сигнализатор об ошибках
        while True:
            pass
    return result


def create_dirs(img_paths, y, logger):
    dest_paths = []
    dirs = retry_connection(list, logger, args=[y.listdir, [upload_dir]])
    for img_path in img_paths:
        if path2str_date(img_path) not in [i['name'] for i in dirs]:
            retry_connection(y.mkdir, logger, args=[upload_dir + path2str_date(img_path)])
            dirs.append(upload_dir + path2str_date(img_path))
        dest_paths.append(upload_dir + path2str_date(img_path)+'/'+img_path.split('/')[-1])
    return dest_paths


def copy_to_yadisk(img_paths, logger):

    y = yadisk.YaDisk(token=ya_token)
    img_paths = json.loads(img_paths)
    dest_paths = create_dirs(img_paths, y, logger)
    for i, dest_path in enumerate(dest_paths):
        #print(dest_path)
        if os.path.isfile(img_paths[i]):
            retry_connection(y.upload, logger, args=[img_paths[i], dest_path])
            logger.debug(img_paths[i]+' copied to '+dest_path)
            while True:
                try:
                    os.remove(img_paths[i])
                    logger.debug(img_paths[i]+' deleted')
                    break
                except PermissionError as e:
                    logger.error(e)
                except FileNotFoundError as e:
                    logger.error(e)
                    break
                time.sleep(10)
        else:
            logger.error(img_paths[i]+" doesn't exists!")
