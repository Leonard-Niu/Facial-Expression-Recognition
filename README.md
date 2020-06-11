# A Pytorch Implementation of FER(Facial-Expression-Recognition)

## Introduction
This project aims to classify facial expression. Here I provide seven types of expression, including  *Angry Disgusted Fearful Happy Sad Surprised Neutral*.
With 250 epochs, this accuracy of baseline achieves **70.382%**

Here is the trained model link: ———— [Trained Model Link](https://pan.baidu.com/s/18YSCZkg1rkRBXaSK5ehVxA) password:5nfw
* Backbone ——VGG16
* Dataset ——FER2013: 

[Dataset Link 240×240 Data（Train、Val、Test）](https://pan.baidu.com/s/1sJOcJR7dcS3xnmFDj24Q_A) password:5j3x
![Backbone](https://github.com/Leonard-Niu/Facial-Expression-Recognition/blob/master/R/VGG-NET.png)

### Highlight
* In this project, **face detection part is applied,** which can definitely improve the test accuracy. More over, it can support the robust of the model, **especially no face input image**.
* **GPU and CPU all support**. If no GPU, it's OK.
* Dependencies fewer.
* When testing, batch images input is supported in the demo.

## Results Show
![Result1](https://github.com/Leonard-Niu/Facial-Expression-Recognition/blob/master/R/2.jpg-result.jpg)
![Result2](https://github.com/Leonard-Niu/Facial-Expression-Recognition/blob/master/R/3.jpg-result.jpg)
## Requirement
**Recommend to use Anaconda**
* Ubuntu16.04 (Windows also avaliable)
* Python 3.6
* Pytorch (latest version or old version are all fine)
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
    ---------------------------00000.jpg
    ---------------------------00005.jpg
    ...
    ------------------ 1
    ---------------------------00023.jpg
    ...
    ...
    ------------------ 6
    ---------------------------00061.jpg
    ...
    
    ------- val
    ------------------ 0
    ---------------------------00006.jpg
    ...
    ------------------ 1
    ---------------------------00043.jpg
    ...
    ...
    ------------------ 6
    ---------------------------00021.jpg
    ...
    
    ------- test
    ------------------ 0
    ---------------------------00008.jpg
    ...
    ------------------ 1
    ---------------------------00011.jpg
    ...
    ...
    ------------------ 6
    ---------------------------00022.jpg
    ...
0-6 represent 7 different expression: ***Angry Disgusted Fearful Happy Sad Surprised Neutral***
## Demo
### Image Input

    python demo_image.py
Running the demo, first need to type the image name, such as *1.jpg*.

Put input images in *input* folder 
### Camera Detection

    python demo_camera.py
### Batch Image Input

    python demo_image_batch.py
## TODO
~~Find image process methods to improve the accuracy.~~

Apply RPN face detection to improve accuracy.
## Issue and Suggestion
Any questions, open a new issue.

### If helpful, please give me a star      
## Reference
* [WuJie1010/Facial-Expression-Recognition.Pytorch](https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch)

* [xionghc/Facial-Expression-Recognition](https://github.com/xionghc/Facial-Expression-Recognition)
