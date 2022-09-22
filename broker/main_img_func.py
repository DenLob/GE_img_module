import gc
import os
import psutil
import json
import time

from PIL import Image

from broker.functions import send
from broker.pika_consts import queue_ya_disk, broker_password_img_work, broker_user_img_work
from constants import CONST_MEDIA_FOLDER
from database.insert_update_funcs import update_pallet
from database.models import create_sessionmaker
from img_stitch.functions import im_stitching
from os.path import isfile

from prepare_imgs.functions import min_img_num, create_final_dict, list_4_remove
from work_with_stitched_img.functions import create_db_info

previous_send_paths = []


def process_images(message, logger):
    proc = psutil.Process(os.getpid())
    mem0 = proc.memory_info().rss / (1024 * 1024)
    global previous_send_paths

    logger.info('Before preparing ' + str(mem0) + ' mb')
    need_key_path = message
    need_key = need_key_path.split('/')[-1].split('_')[0]
    all_paths = list_4_remove(need_key_path)
    previous_send_paths_tmp = all_paths
    all_paths = list(filter(lambda path: path not in previous_send_paths, all_paths))
    previous_send_paths = previous_send_paths_tmp
    send(data=json.dumps(all_paths), queue_name=queue_ya_disk, user=broker_user_img_work,
         password=broker_password_img_work, logger=logger)
    del all_paths
    gc.collect()
    while True:
        if isfile(need_key_path):
            logger.info(message + ' downloaded')
            break
        else:
            time.sleep(30)
    logger.info('WORKING_ON ' + need_key)

    final_dict, debug_info = create_final_dict(need_key_path)
    mem0 = proc.memory_info().rss / (1024 * 1024)
    logger.info('After preparing ' + str(mem0) + ' mb')
    logger.info('CREATED_DICT_FOR ' + need_key)
    no_pallets = True

    for key in final_dict:
        if len(final_dict[key]) >= min_img_num and need_key in key:
            no_pallets = False
            new_path = im_stitching(final_dict[key], key, debug_info, logger=logger)
            mem0 = proc.memory_info().rss / (1024 * 1024)
            logger.info('After stitching ' + str(mem0) + ' mb')
            im_pil = Image.open(new_path)
            fixed_height = 300
            height_percent = (fixed_height / float(im_pil.size[1]))
            width_size = int((float(im_pil.size[0]) * float(height_percent)))
            im_pil = im_pil.resize((width_size, fixed_height))
            im_rotate = im_pil.rotate(90, expand=True)
            im_rotate.save(CONST_MEDIA_FOLDER + 'thumb_' + new_path.split('/')[-1], quality=95)
            im_pil.close()
            del im_pil
            db_data = create_db_info(img_path=new_path, key=key, need_key=need_key, logger=logger)
            sessionmaker = create_sessionmaker()
            update_pallet(data=db_data, Session=sessionmaker)
            del db_data
            gc.collect()
    del final_dict
    del debug_info
    if no_pallets:
        logger.info('NO_PALLETS_FOR ' + need_key)
    gc.collect()
    mem0 = proc.memory_info().rss / (1024 * 1024)
    logger.info('After full process ' + str(mem0) + ' mb')

