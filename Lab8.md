# Lab8 Introduction to computer vision

## Computer Vision
 - computer vision is a field of artificial intelligence(AI) that enables computers and asy systems to drive meaningful information from digital images, videos and other visual inputs - and take actions or make recommendations based on that information
 - If AI enables computers to think, computer vision enables them to see, observe and understand

 1. Computers can see the world through digital images, such as photographs, live camera feeds, or recorded video files
 2. Each digital image is actually an array of numeric values indicating the color intensity of each pixel in the image. You can use these values just like any other data as feature to train a machine learning model, or to apply transformation and calculations to analyze the image data and find patterns in it
 3. A grid of numbers is overlaid on the image - each number is a value between 0 and 255, representing the intensity of each pixel
 4. In the case of color images, there  are three layers of pixel values for red, green and blue intensities.(Three overlapping tables of red, green and blue pixel values)

![](/Lab8/Picture1.png)

## Applications of Computer Vision

 - Image Classification
 - Object Detection
 - Semantic Segmentation
 - Image Analtsis
 - Face Detection & Recognition
 - Optical Character Recognition

## Image Classification

 - An image classification model uses the unmeric pixel features to predict a class label for the image
 - For example, in a traffic monitoring solution you might use an image classification model to classify images based on the type of vehicle they contain, such as taxis, buses, cyclist, and so on. Essentinally, image classification models answer the question "What is this an image of?"

## Object Detection 
 - Object detection models are teained to classify individual objects in the image, and to indicate their location within the image as a bounding box
 - For example, you could expand the traffic monitoring sulution to detect the presence of multiple classes of vehicle in an image. Object detection models are used to answer the question "What object are in this iamges, and where are they"

## Semantic Segmentation

 - Semantic segmentation models are an advance kind of model taht classify individual pixels in the iamge based on which object they belong to
 - For example, the traffic monitoring might use a semantic segmentation model to overlay "masks" on the iamge to indicate pixels that belong to buses, cars, and cyclists with different colors. Semantic segmentation answers the question "Which pixels belong to which object?"


## Image Analysis

 - You can build on object detection and classification models to perform image analysis, generating captions and tags based on the image contents

## Face Detection & Recognition

 - Face detection is used to identify the location of human faces in an image. Further analysis can be perfored on facial features to determine facial features, emotion, age, and so on; and models can be tried to identify specific individuald based on facial recognition

## Optical Character Recognition
 - An OCR model indentifies regions of the image that contain text and ertracts them

## OpenCV
### What is OpenCV
 - is an open-couse package and hence it's free for both academic and commercial use
 - has C++, C, Python and java interfaces and supports Window, Linux, MacOS, IOS and Android
 - designed for computational efficiency and with a strong focus on real-time applications
 - Written in optimized C/C++, the library can take advantage of multi-core processing

## Images and Pixels
 - An image consists of some picture elements, called pixels ot pixel values
 - In the case of color images, we hace three colored channels
 - Hence colored images will have multiple values for single-pixel values
 - The color values go from 0 to 255
 - These color channels are generally represented as Red, Green Blue(RGB) for instance

## Basic Operations

```py
# Load and Show An Image
import cv2
import matplotlib.ptplot as plt

img = cv2.imread('images/puppy.jpg')
plt.show(img)
```
![](/Lab8/Picture2.png)

```py
import cv2
img = cv2.imread('images/puppy.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BRG2RGB)
plt.imshow(img_rgb)
```
![](/Lab8/Picture3.png)

```py
import cv2
img_rgb = cv2.cvtColor(img, cv2.COLOR_BRG2RGB)
Img_rs = cv2.resize(igm_rgb, (900,275))
plt.imshow(img_rs)
```
![](/Lab8/Picture4.png)

```py
img_rgb = cv2.cvtColor(img, cv2.COLOR_BRG2RGB)

# Along central x-axis
new_img = cv2.flip(igm_rgb, 0)
plt.imshow(new_img)

# Along central y-axis
new_img = cv2,flip(img_rgb, 1)
plt.imshow(new_img)
```
![](/Lab8/Picture5.png){width=450 px;}

```py
import cv2
import matplotlib.pyplot as plt
img1 = cv2.imread('images/puppy2.jpg')
img2 = cv2.imread('images/do_not_copy.png')
img1 = cv2.cvtColot(img1, cv2.COLOR_BRG2RGB)
img2 = cv2.cvtColot(img2, cv2.COLOR_BRG2RGB)
img1 = cv2.resize(img1, (1200, 1200))
img2 = cv2.resize(img2, (1200, 1200))
blended = cv2.addWeighted(sec1 = img1, alpha = 0.7, src2 = img2, beta = 0.3, gamma = 0)

plt.figure(figsize = (8,8))
plt.imshow(blended)
```
![](/Lab8/Picture6.png)

```py
import numpy as np
import cv2
import matplotlib.pyplot as plt
%matplotlob inline

def display_img(img):
    fig = plt.figure(figsize = 8,6)
    as = fig.ass_subplot(111)
    as.imshow(img, cmap = 'gray')

# read the image
image = cv2.imread('images/puppy2.jpg')

# calcualte the edgesusing Cannt edge algorithm
edges = cv2.Canny(image, 100, 200)

# plot the edges
```
![](/Lab8/Picture7.png)

```py
# face detection
import numpy as np
import cv2
import matplotlib.pyplot as plt

smile = cv2.imread('images/smile.jpg', 0)

# OpenCV comes with these pre-trained cascade files
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

def detect_face(img):
    face_img = img.copy()
    face_rects = face_cascade.detectMultiScale(face_img)

    for(x,y,w,h) in face_rects:
        cv2.rectangle(face_img, (x,y), (x+w, y+h), (255, 255, 255), 3)
    return face_img

result = detect_face(smile)
plt.imshow(result, cmap = 'gray')
```
![](/Lab8/Picture8.png)

## Models - YOLO

 - YOLO(You Only Look Once) is an object detection algorithm that can be implemented in Python. It is designed to detect objects in images or video frames, providing bounding box corrdinates and class labels for each detected object
 - originlly developed by Joseph Redmon and Ali Farhadi at the University of Washington in 2015

## Models - Detectron2

 - "Detectron2 is Facebook AI Research's next generation library that provides state-of-art detection and segmentation algorithms. It is the successot of Detectron and maskrcnn-benchmark, It supports a number of computer vision research projects and production applications in Facrbook"