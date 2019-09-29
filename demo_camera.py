#----------------------------------------
# written by Leonard Niu
# HIT
#----------------------------------------
import numpy as np
import os
import argparse
from model import VGG16

from PIL import Image
import cv2
from utils import camera_face_detect

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms


parser = argparse.ArgumentParser(description='Facial Expression Recognition')
parser.add_argument('--input_size', type=int, default=224, help='Net')
parser.add_argument('--SAVE_FLAG', default=False, help='result save or not')
parser.add_argument('--output_folder', default='./camera_output')
parser.add_argument('--ckpt_path', default='./checkpoints/FinalModel.t7')

cfg = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

transform_test = transforms.Compose([
    transforms.TenCrop(cfg.input_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])


def camera_predict():

    video_captor = cv2.VideoCapture(0)
    predicted_class = None
    while True:
        ret, frame = video_captor.read()

        face_img, face_coor = camera_face_detect.face_d(frame)

        if face_coor is not None:
          [x_screen,y_screen,w_screen,h_screen] = face_coor
          cv2.rectangle(frame, (x_screen,y_screen), (x_screen+w_screen,y_screen+h_screen), (255,0,0), 2)

        if cv2.waitKey(1) & 0xFF == ord(' '):
            face_img, face_coor = camera_face_detect.face_d(frame)

            if face_coor is not None:
                [x,y,w,h] = face_coor
                cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

            if face_img is not None:
                if not os.path.exists(cfg.output_folder):
                    os.mkdir('./camera_output')
                cv2.imwrite(os.path.join(cfg.output_folder, 'face_image.jpg'), face_img)
                gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
                gray = cv2.resize(gray, (240, 240))
                img = gray[:, :, np.newaxis]
                img = np.concatenate((img, img, img), axis=2)
                img = Image.fromarray(np.uint8(img))
                inputs = transform_test(img)
                class_names = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Sad', 'Surprised', 'Neutral']

                net = VGG16.Net()
                checkpoint = torch.load(cfg.ckpt_path)
                net.load_state_dict(checkpoint['net'])

                if use_cuda:
                    net.to(device)

                net.eval()
                ncrops, c, h, w = np.shape(inputs)
                inputs = inputs.view(-1, c, h, w)
                inputs = Variable(inputs, volatile=True)
                if use_cuda:
                    inputs = inputs.to(device)
                outputs = net(inputs)
                outputs_avg = outputs.view(ncrops, -1).mean(0)
                score = F.softmax(outputs_avg)
                print(score)
                _, predicted = torch.max(outputs_avg.data, 0)
                predicted_class = class_names[int(predicted.cpu().numpy())]
                print(predicted_class)

            if predicted_class is not None:
                cv2.putText(frame, predicted_class, (30, 60),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 1)
                cv2.imwrite(os.path.join(cfg.output_folder, 'predict.jpg'), frame)


        if predicted_class is not None:
            cv2.putText(frame, predicted_class, (30, 60),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 1)
        cv2.imshow('camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    camera_predict()