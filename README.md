# A Pytorch Implementation of FER(Facial-Expression-Recognition)

## Introduction
This project aims to classify facial expression. Here I provide seven types of expression, including  *Angry Disgusted Fearful Happy Sad Surprised Neutral*.
* Backbone ——VGG16
* Dataset ——FER2013(If you need the dataset, please email me)

### Highlight
* In this project, **face detection part is applied,** which can definitely improve the test accuracy. More over, it can support the robust of the model, **especially no face input image**.
* **GPU and CPU all support**. If don't have GPU, is OK.
* Dependencies fewer.
* when testing, batch images input is supported.


## Requirement
**Recommend to use Anaconda**
* Ubuntu16.04 (Windows also avaliable, but need to change something, like image path)
* Python 3.6
* Pytorch (latest version or old version all fine, mine is 0.4.1 & 1.1.0)
* torchvision
* numpy
* matplotlib
* opencv(cv2)
* pillow

## Dataset Process
FER2013 includes 35887 pictures: 48 × 48 pixels, here using bilinear interpolation to resize the expression pictures to 240 × 240 pixels.
**The input of the net is 224 × 224**, same as original VGG16.

## Train
First, put the processed dataset in the folder "data", the data folder like following:

    -- data
    ------- train
    ------------------ 0
    ------------------ 1
    ...
    ------------------ 6
    
    ------- val
    ------------------ 0
    ------------------ 1
    ...
    ------------------ 6
    
    ------- test
    ------------------ 0
    ------------------ 1
    ...
    ------------------ 6
0-6 represent 7 different expression:Angry Disgusted Fearful Happy Sad Surprised Neutral
## Demo
### Image Input

    python demo_image.py
When running, first need to type the image name, such as *1.jpg*.
Put input images in *input* folder 
### Camera Detection

    python demo_camera.py
### Batch Image Input

    python demo_image_batch.py
## TODO
Find image process methods to improve the accuracy.
## Issue and Suggestion
Any questions, open a new issue.

### If you think it's helpful, please give me a star      ^ _ ^
## Reference
* [WuJie1010/Facial-Expression-Recognition.Pytorch](https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch)

* [xionghc/Facial-Expression-Recognition](https://github.com/xionghc/Facial-Expression-Recognition)
