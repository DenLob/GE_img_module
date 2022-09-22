import cv2

from neuron.script import get_topn_pred


def identify(img_path, boxes):
    identified_plants = []
    if len(boxes) != 0:
        img = cv2.imread(img_path)
        for i, box in enumerate(boxes):
            box = box['center']
            img_crop = img[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
            img_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
            defined_type = get_topn_pred(img_crop)
            identified_plants.append(defined_type)
    return identified_plants
