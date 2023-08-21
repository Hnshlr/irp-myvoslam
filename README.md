# AY22 Individual Research Project (A) 2022-2023

Research Project on Autonomous Vehicle Self Awareness using SLAM algorithms.

This project is research as part of the Master Thesis for the MSc in Computational and Software Techniques at Cranfield University.

It aims to provide an analysis of the extent to which Visual Odometry can achieve accurate position and orientation estimation of a vehicle in complex environments.

© Copyright 2023, All rights reserved to Hans Haller, CSTE-CIDA Student at Cranfield Uni. SATM, Cranfield, UK.

‎ 

## Setting up the application

### Python version

The application was initially developed with Python 3.7.17, however in the late stages of development, a proper virtual environment was created under Python 3.9.17. In both cases, make sure you use respectively the ```requirements37.txt``` or ```requirements39.txt``` file to install the dependencies, depending on the version of Python you wish to use.

### Important notes

If Python 3.9.17 is used, you won't be able tu use ```SURF``` as a feature detector, as it is not supported by OpenCV 4.5.4, the lowest version of OpenCV that supports Python 3.9.17.

### Initiate the virtual environment

Once the repository of the project is cloned, or downloaded from Cranfield's Canvas, create a virtual environment for the project to store all packages the app requires. 

To create a virtual environment, first open your command prompt or terminal and navigate to the main directory:

```cd ./irp-myvoslam```

Once in the main directory, create a new virtual environment using the venv module in Python. The command to create a virtual environment is:

```python3.9 -m venv venv``` (or ```python3.7 -m venv venv``` if you are using Python 3.7.17)

This will create a new directory called "venv" in the backend directory, which will contain all the necessary files for their virtual environment. 

To activate the virtual environment, run the following command:

```source venv/bin/activate```

This will activate the virtual environment. You should see the name of the environment in your terminal prompt. Once the virtual environment is activated, you can install the dependencies required for the project using pip. The list of the required packages is listed in the requirements.txt file in the root of the directory backend.

To install the dependencies, run the following command:

```pip3 install -r requirements39.txt --no-cache-dir --no-dependencies``` (or ```pip3 install -r requirements37.txt --no-cache-dir --no-dependencies``` if you are using Python 3.7.17)

This will install all the required packages for the project.

### Download the data

To download the data, run the following command:

```./scripts/curl-data-and-models.sh```

or 

```./scripts/wget-data-and-models.sh```

This will download the data from the KITTI dataset, create the ```src/data/input/kitti``` and ```src/data/output/kitti``` directories, and unzip the data in the input directory. It will also download the pre-trained models for the feature detectors and descriptors, and unzip them in the ```src/data/models``` directory. You must have either ```curl``` or ```wget``` installed on your machine for this to work.


### Run the application

Once the virtual environment is now set up and ready to use, the user can start the application by running the app.py file through their IDE, or using the following command:

```python3 main.py```

### You're set to go!

If you have any questions, please contact me at ```hans.haller.885@cranfield.ac.uk```

‎ 

## Running specific tasks

The application is designed to be modular, and to allow the user to run specific tasks. The following sections will describe how to run specific tasks.

### Compute the pose estimation

...

### Visualize the features, matches and more

...

### Bulk-Testing of FTO grid combinations

...


## Contact:

GitHub: https://www.github.com/Hnshlr

LinkedIn: https://www.linkedin.com/in/hans-haller/

