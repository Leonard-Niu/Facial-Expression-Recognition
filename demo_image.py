#-----------------------------------------
# written by Leonard Niu
# HIT
#-----------------------------------------
import numpy as np
import os
import argparse

import matplotlib.pyplot as plt
from PIL import Image
import cv2
from utils import image_face_detect

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from model import VGG16


parser = argparse.ArgumentParser(description='Facial Expression Recognition')
parser.add_argument('--input_size', type=int, default=224, help='Net')
parser.add_argument('--SAVE_FLAG', default=False, help='result save or not')
parser.add_argument('--input_folder', default='./input')
parser.add_argument('--output_folder', default='./image_output')
parser.add_argument('--ckpt_path', default='./checkpoints/FinalModel.t7')
arg = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

transform_test = transforms.Compose([
    transforms.TenCrop(arg.input_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])





def image_predict():
    image_name = str(input("Image name:"))
    image_path = os.path.join(arg.input_folder, image_name)
    src_img = np.array(Image.open(image_path))
    face_img, face_coor = image_face_detect.face_d(src_img)
    gray = cv2.cvtColor(face_img,cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (240, 240))
    img = gray[:, :, np.newaxis]
    img = np.concatenate((img, img, img), axis=2)
    img = Image.fromarray(np.uint8(img))
    inputs = transform_test(img)

    class_names = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Sad', 'Surprised', 'Neutral']

    net = VGG16.Net()
    checkpoint = torch.load(arg.ckpt_path)
    net.load_state_dict(checkpoint['net'])
    if use_cuda:
        net.cuda()
    net.eval()
    ncrops, c, h, w = np.shape(inputs)
    inputs = inputs.view(-1, c, h, w)

    inputs = Variable(inputs, volatile=True)
    if use_cuda:
        inputs = inputs.to(device)
    outputs = net(inputs)
    outputs_avg = outputs.view(ncrops, -1).mean(0)
    score = F.softmax(outputs_avg)
    _, predicted = torch.max(outputs_avg.data, 0)
    expression = class_names[int(predicted.cpu().numpy())]
    if face_coor is not None:
        [x,y,w,h] = face_coor
        cv2.rectangle(src_img, (x,y), (x+w,y+h), (255,0,0), 2)

    plt.rcParams['figure.figsize'] = (11,6)
    axes=plt.subplot(1, 2, 1)
    plt.imshow(src_img)
    plt.title('Input Image', fontsize=20)
    axes.set_xticks([])
    axes.set_yticks([])
    plt.tight_layout()
    plt.subplots_adjust(left=0.05, bottom=0.2, right=0.95, top=0.9, hspace=0.02, wspace=0.3)
    plt.subplot(1, 2, 2)
    ind = 0.1+0.6*np.arange(len(class_names))
    width = 0.4
    for i in range(len(class_names)):
        plt.bar(ind[i], score.data.cpu().numpy()[i], width, color = 'orangered')
    plt.title("Result Analysis",fontsize=20)
    plt.xticks(ind, class_names, rotation=30, fontsize=12)
    if arg.SAVE_FLAG :
        if not os.path.exists(arg.output_folder):
            os.mkdir('./image_output')
        save_path = os.path.join(arg.output_folder, image_name)
        plt.savefig(save_path + '-result' + '.jpg')
    else:
        plt.show()


    print("The Expression is %s" %expression)



if __name__ == '__main__':
    image_predict()