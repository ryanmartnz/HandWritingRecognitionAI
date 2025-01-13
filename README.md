# Hand-Writing Recognition AI in Python

## Project Overview

This project receives an image of handwritten words on a blank page, detects the letters in the image,
and predicts what english letters are present on the page. The program will output each word detected in 
the image, separated by a white space. Different lines in the image are also separated in the output.

## Setting Up and Running the Project

1)  Unzip the contents of the "demo" zip and navigate to the main directory
    ./demo in a terminal that can utilize python.

2) Create a python venv within the main directory ./demo and activate it using the following commands

    $ python3.12 -m venv .  
    $ source ./bin/activate  

3)  Run the following command within the main directory ./demo
    to install all of the required python dependencies within venv.

    $ pip install -r requirements.txt

4)  The demo inputs are are within the ./demo_inputs directory.
    You can view them there.

    To view the output, run the python file 'writing_recognition.py' within the main directory. 
    The program will ask for input on which demo image to use.
    Enter either "./demo_inputs/test1.jpg", "./demo_inputs/test2.jpg", or "./demo_inputs/test3.jpg"
    
    After the program finishes running, it will print to the console the words detected on the image.

    $ python writing_recognition.py


### Files Included:

1) writing_recognition.py
    This is the main program. It detects the letters in the image and uses the model 
    saved in "model.pth" to predict what letters are on the image.

2) model.py
    This is the program that contains the definition of the CNN model used in writing_recognition.py.
    It trains the model and evaluates its accuracy over epochs. It attempts to maximize the accuracy by 
    resetting the number of epochs once a new best accuracy has been achieved. For more information,
    read the comments provided in the file.

3) model.pth
    This file contain the parameters and buffers of the trained model obtained from model.py.
    This file is loaded into the CNN model used in writing_recognition.py.

4) requirements.txt
    This file holds all of the python dependencies required to run this project.


### Directories:

1) demo
    This directory is the main directory which holds the python files and 
    subdirectories needed to run the project.

2) demo/train_model
    This directory holds the input images for the model to train on.
    The images in the demo/train_model/alphabet directory were derived from the image demo/train_model/training1.png
    The images in the demo/train_model/letters directory were derived from the image demo/train_model/training2.png

3) demo/test_model
    This directory holds the images used to evaluate the model.
    Each image is of a letter in the english alphabet, and each image is labeled with its respective letter 
    in the text file matching the name of the image.

4) demo/demo_inputs
    This directory holds the images used to test the model.
    These images are not labeled, and they are images of handwritten words on a blank page.
    These images are for use in writing_recognition.py.
