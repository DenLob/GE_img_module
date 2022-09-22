from work_with_stitched_img.count_functions import count_plants
from work_with_stitched_img.identify_plant import identify


def create_db_info(img_path, key, need_key, logger):
    plant_centers = [{'center': center, 'type_plant': None} for center in count_plants(img_path, logger=logger)]
    plant_types = [{'type_plant': None, 'count': 0}]
    identified_plants = identify(img_path, plant_centers)
    for i, plant_center in enumerate(plant_centers):
        plant_center['type_plant'] = identified_plants[i]
        tmp = list(filter(lambda item: item['type_plant'] == plant_center['type_plant'], plant_types))
        if len(tmp) != 0:
            plant_types[plant_types.index(tmp[0])]['count'] += 1
        else:
            plant_types.append({'type_plant': plant_center['type_plant'], 'count': 1})
    max_count = 0
    most_plant_type = ''
    for i in plant_types:
        if i['count'] > max_count:
            max_count = i['count']
            most_plant_type = i['type_plant']
    db_data = {key:
        {
            'id': None,
            'time': need_key,
            'path': img_path,
            'plants': plant_centers,
            'type_plant': most_plant_type
        }
    }
    return db_data
