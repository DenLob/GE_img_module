import cv2
import torch
from torchvision import transforms
from PIL import Image

crop_flag = True

def load_image(cv_img):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img = Image.fromarray(cv_img)
    img = transform(img)
    img = img.reshape(1, 3, 256, 256)
    return img


def load_class_names(filename):
    with open(filename) as f:
        return f.read().splitlines()


def get_predict(img, model, class_names, device):
    pred_list = []
    model.eval()
    img = load_image(img)
    pred_list = model(img).sort(descending=True)[1].tolist()
    return pred_list[0]


def get_topn_pred(img):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if crop_flag:
        model = torch.jit.load('model_croped.pt')
    else:
        model = torch.jit.load('model.pt')
    predictions = get_predict(img,
                              model,
                              'classes.txt',
                              device)

    class_names = load_class_names('classes.txt')

    top_pred = class_names[predictions[0]]

    # print(top_pred)
    return top_pred



