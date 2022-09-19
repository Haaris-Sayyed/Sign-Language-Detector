
<p align="center">
<img src="img/banner.png" alt="banner" width="90%">
</p>

<div align="center">

<strong>A python based app which can convert the hand gesture to corresponding text label in real time.</strong>
 
<a href="">![GitHub repo size](https://img.shields.io/github/repo-size/Haaris-Sayyed/Sign-Language-Detector?style=for-the-badge&color=important)</a> &nbsp;&nbsp;
<a href="">![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg?style=for-the-badge&color=important)</a> 
</div>
<p align="center">
<img src="img/SignDetector.gif" width="80%">
</p>

***

## Basic Overview

<p align="center"><strong>" Talk to a man in a language he understands, that goes to his head. Talk to him in his own language, that goes to his heart. "</strong></p> 
<p>This project introduces Sign Language recognition system which can recognize American Sign Language.
</p><br>
<p align="center">
<img src="img/american_sign_language.png" width="80%">
</p>
<p>The project makes use of <strong>Convolutional Neural Network (CNN) Algorithm</strong> for training and to classify the images. The proposed system was able to <strong>recognize 10 American Sign gesture alphabets </strong> with high accuracy.</p>
<p>Model has achieved a remarkable <strong>accuracy of 97.89%</strong></p>
<p align="center">
<img src="img/accuracy.png">&nbsp;&nbsp;
<img src="img/loss.png">
</p>

--- 

## Getting Started

It requires python version 3.6 or later as to synchronize with tensorflow.

- **collect_data.py**  file will help in creating your own dataset using webcam.

- **cnn_model.py** file will use Convolutional Neural Network (CNN) to train the model and store it in the form of hadoop distributed (h5) format.

- **gesture_predict.py** file will recognise the gesture as per the trained dataset.

- **SignDetector.py** file contains the code used to built the UI of the application.
---

## Pre-requisites

- python 3.6 or later
- tkinter module
- tensorflow
- keras
- opencv
- pillow
- numpy
- imutils
- matplotlib

---

## Installations

### 1. Install python 3 
Python is usually installed by default on most modern systems. To check what python version you currently have, open a terminal and run the following command:
```
python --version
```

   This should output some information on the installed Python version. You can also install python by following these instructions:
   [https://installpython3.com/](https://wsvincent.com/install-python/)

### 2. Install tensorflow framework

Install tensorflow framework with the following command
```
python -m pip install tensorflow --trusted-host files.pythonhosted.org --trusted-host pypi.org --trusted-host pypi.python.org
```
For upgradation use the following command

``` 
pip install tensorflow==2.0.0-alpha0
```

### 3. Install keras

Install keras with the following command
```
pip install keras
```
**Note:** if keras doesnt work just replace **keras.model** to **tensorflow.keras.model** and **keras.preprocessing** to **tensorflow.keras.preprocessing** 

### 4. Install OpenCV 

Install opencv for python with the following commands

```
pip install opencv-python==3.4.2.16
```

```
pip install opencv-contrib-python==3.4.2.16
```

### 5. Install pillow

Install PIL with the following command

```
pip install pillow
```

### 6. Install numpy 

Install numpy with the following command

```
pip install numpy
```

### 7. Install imutils 

Install imutils with the following command

```
pip install imutils
```

### 8. Install matplotlib 

Install matplotlib with the following command

```
pip install matplotlib
```

---

## How to run the project ?

There are 2 ways to run the project which are as follows:

1. You will be able to run the project by simply running the **SignDetector.py** file.

2. You will be able to execute the app by running the following command from terminal in the project directory.

<strong>On Windows:</strong>

```
.\SignDetector.bat
```

<strong>On Unix:</strong>

```
./SignDetector.sh
```
---
<p align="center">
<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=32&pause=1000&color=010103&background=246C53&center=true&vCenter=true&width=435&lines=C%3A%3E_+Open+Source%E2%9D%A4%EF%B8%8F+" alt="Typing SVG" /></a></p>

---